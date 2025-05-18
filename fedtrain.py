import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger
from Trainer.kmeansplus_trainer import FedtrainTrainer
from model.FFMSR_fed import MLTRec
# from model.ablation.no_gate import MLTRec
from data.dataset import ECGDataset


# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def finetune(dA, dB, pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/FMETF.yaml', 'props/pretrain.yaml']
    print(props)

    # configurations initialization
    config_A = Config(model=MLTRec, dataset=dA, config_file_list=props, config_dict=kwargs)
    config_B = Config(model=MLTRec, dataset=dB, config_file_list=props, config_dict=kwargs)
    config_C = Config(model=MLTRec, dataset=dC, config_file_list=props, config_dict=kwargs)

    init_seed(config_A['seed'], config_A['reproducibility'])
    init_seed(config_B['seed'], config_B['reproducibility'])
    init_seed(config_B['seed'], config_C['reproducibility'])

    # logger initialization
    init_logger(config_A)
    init_logger(config_B)
    init_logger(config_C)
    logger = getLogger()
    logger.info(config_A)
    logger.info(config_B)
    logger.info(config_C)

    # dataset filtering
    dataset_A = ECGDataset(config_A)
    logger.info(dataset_A)
    dataset_B = EEG(config_B)
    logger.info(dataset_B)
    dataset_C = PPG(config_C)
    logger.info(dataset_C)


    # dataset splitting
    train_data_A, _, _ = data_preparation(config_A, dataset_A)
    train_data_B, _, _ = data_preparation(config_B, dataset_B)
    train_data_C, _, _ = data_preparation(config_C, dataset_C)

    # model loading and initialization
    model_A = MLTRec(config_A, train_data_A.dataset).to(config_A['device'])
    model_B = MLTRec(config_B, train_data_B.dataset).to(config_B['device'])
    model_C = MLTRec(config_C, train_data_C.dataset).to(config_C['device'])


    # trainer loading and initialization
    trainer = FedtrainTrainer(config_A, config_B, config_C, model_A, model_B, model_C)
    trainer.fedtrain(train_data_A, train_data_B, train_data_C show_progress=True)


    return config_A['model'], config_A['dataset']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dA', type=str, default='ECG', help='dataset name')
    parser.add_argument('-dB', type=str, default='EEG', help='dataset name')
    parser.add_argument('-dC', type=str, default='PPG', help='dataset name')

    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=False)
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.dA, args.dB, args.dC, pretrained_file=args.p, fix_enc=args.f)
