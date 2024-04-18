import config
from core import dataloading, models, utils, tracking
import pdb

class OSMIS_Generator():
    def __init__(self, epoch_num, run_name="corrosion_dataset", num_generated=100):
        # --- read options --- #
        self.opt = config.read_arguments(train=False, preset_args=["--exp_name", run_name, f"--which_epoch={epoch_num}",f"--num_generated={num_generated}"])

        # --- create dataloader and recommended model config --- #
        _, self.model_config = dataloading.prepare_dataloading(self.opt)

        # --- create models, losses, and optimizers ---#
        self.netG, _, self.netEMA = models.create_models(self.opt, self.model_config)

        # --- create utils --- #
        self.visualizer = tracking.visualizer(self.opt)

    def get_batch_of_images(self):
        # --- generate images and masks --- #
        for i in range(self.opt.num_generated):
            z = utils.sample_noise(self.opt.noise_dim, 1).to(self.opt.device)
            fake = self.netEMA.generate(z) if not self.opt.no_EMA else self.netG.generate(z)
            self.visualizer.save_batch(fake, self.opt.continue_epoch, i=str(i),return_image=False)

# test = OSMIS_Generator(epoch_num=20000,run_name="corrosion_dataset", num_generated=200)
# test.get_batch_of_images()