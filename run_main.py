import argparse
from omegaconf import OmegaConf
from utils.c_generate_pipeline import GeneratePipeline

def main(config):
    gp = GeneratePipeline(config=config)
    final_res = gp.process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/config_v1.yaml', type=str)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)

