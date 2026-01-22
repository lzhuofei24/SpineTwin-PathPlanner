import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from src.models.srgan import SRGenerator, SRDiscriminator


class SRTrainer(pl.LightningModule):
    def __init__(self, lr=1e-4, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()

        # 1. 初始化模型
        self.generator = SRGenerator()
        self.discriminator = SRDiscriminator()

        # 2. 关键修复：关闭自动优化，开启手动优化模式 (适配 GAN)
        self.automatic_optimization = False

        # 损失函数权重
        self.lambda_adv = 1e-3  # 对抗损失权重
        self.lambda_pixel = 1.0  # 像素损失权重

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        # 3. 手动获取优化器
        opt_g, opt_d = self.optimizers()

        imgs_lr = batch["lr"]
        imgs_hr = batch["hr"]

        # =========================
        #  训练生成器 (Generator)
        # =========================
        # 切换到生成器优化器上下文
        self.toggle_optimizer(opt_g)

        # 生成高分图
        gen_hr = self(imgs_lr)

        # A. Pixel Loss (L1/MSE): 保证内容结构一致
        loss_pixel = F.mse_loss(gen_hr, imgs_hr)

        # B. Adversarial Loss: 希望判别器认为是真的(label=1)
        # 创建全1的标签，并自动匹配设备(GPU/CPU)
        valid = torch.ones(imgs_hr.size(0), 1).type_as(imgs_hr)

        pred_real = self.discriminator(gen_hr)
        loss_adv = F.binary_cross_entropy_with_logits(pred_real, valid)

        # 总生成器 Loss
        loss_G = (self.lambda_pixel * loss_pixel) + (self.lambda_adv * loss_adv)

        # 手动反向传播与更新
        self.manual_backward(loss_G)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # =========================
        #  训练判别器 (Discriminator)
        # =========================
        self.toggle_optimizer(opt_d)

        # A. Real Loss: 判别器看真图，标签是1
        pred_real = self.discriminator(imgs_hr)
        loss_real = F.binary_cross_entropy_with_logits(pred_real, valid)

        # B. Fake Loss: 判别器看假图(detach防止梯度传回G)，标签是0
        fake = torch.zeros(imgs_hr.size(0), 1).type_as(imgs_hr)
        # 注意：这里必须 detach()，否则梯度会传回生成器
        pred_fake = self.discriminator(gen_hr.detach())
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, fake)

        # 总判别器 Loss
        loss_D = (loss_real + loss_fake) / 2

        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # 4. 记录日志 (prog_bar=True 表示在进度条显示)
        self.log("g_loss", loss_G, prog_bar=True)
        self.log("d_loss", loss_D, prog_bar=True)
        self.log("g_pixel_loss", loss_pixel)
        self.log("g_adv_loss", loss_adv)

    def validation_step(self, batch, batch_idx):
        # 5. 补充验证步骤
        imgs_lr = batch["lr"]
        imgs_hr = batch["hr"]
        gen_hr = self(imgs_lr)

        # 验证集通常只看像素还原程度 (PSNR/MSE)
        val_loss = F.mse_loss(gen_hr, imgs_hr)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []