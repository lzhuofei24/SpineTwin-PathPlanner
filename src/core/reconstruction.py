import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from src.models.srgan import SRGenerator, SRDiscriminator


class SRTrainer(pl.LightningModule):
    def __init__(self, lr=1e-4, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()

        self.generator = SRGenerator()
        self.discriminator = SRDiscriminator()
        self.automatic_optimization = False
        self.lambda_adv = 1e-3
        self.lambda_pixel = 1.0

    def forward(self, z):
        return self.generator(z)

    # --- 新增辅助函数：计算 PSNR ---
    def calculate_psnr(self, img1, img2):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return 100
        return 10 * torch.log10(1 / mse)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        imgs_lr = batch["lr"]
        imgs_hr = batch["hr"]

        # 1. 训练生成器
        self.toggle_optimizer(opt_g)
        gen_hr = self(imgs_lr)

        loss_pixel = F.mse_loss(gen_hr, imgs_hr)
        valid = torch.ones(imgs_hr.size(0), 1).type_as(imgs_hr)
        pred_real = self.discriminator(gen_hr)
        loss_adv = F.binary_cross_entropy_with_logits(pred_real, valid)

        loss_G = (self.lambda_pixel * loss_pixel) + (self.lambda_adv * loss_adv)

        self.manual_backward(loss_G)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # 2. 训练判别器
        self.toggle_optimizer(opt_d)
        pred_real = self.discriminator(imgs_hr)
        loss_real = F.binary_cross_entropy_with_logits(pred_real, valid)
        fake = torch.zeros(imgs_hr.size(0), 1).type_as(imgs_hr)
        pred_fake = self.discriminator(gen_hr.detach())
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, fake)
        loss_D = (loss_real + loss_fake) / 2

        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # 3. 记录日志 (新增 train_psnr)
        # 这里的 batch_size=imgs_lr.size(0) 是为了消除之前的警告
        self.log("g_loss", loss_G, prog_bar=True, batch_size=imgs_lr.size(0))
        self.log("d_loss", loss_D, prog_bar=True, batch_size=imgs_lr.size(0))

        # 计算训练集的 PSNR 用于监控
        with torch.no_grad():
            train_psnr = self.calculate_psnr(gen_hr, imgs_hr)
            self.log("train_psnr", train_psnr, prog_bar=True, batch_size=imgs_lr.size(0))

    def validation_step(self, batch, batch_idx):
        imgs_lr = batch["lr"]
        imgs_hr = batch["hr"]
        gen_hr = self(imgs_lr)

        val_loss = F.mse_loss(gen_hr, imgs_hr)
        val_psnr = self.calculate_psnr(gen_hr, imgs_hr)

        # 记录验证集指标
        self.log("val_loss", val_loss, prog_bar=True, batch_size=imgs_lr.size(0))
        self.log("val_psnr", val_psnr, prog_bar=True, batch_size=imgs_lr.size(0))

        return val_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []