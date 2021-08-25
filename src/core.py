import os

import constants


class Core(object):
    def get_args(self):
        # to be implemented in sub-class
        return None

    def get_trainer(self, args):
        # to be implemented in sub-class
        return None

    def run(self):
        ################################
        # Make necessary directories

        if not os.path.exists(constants.LOG_DIR):
            os.mkdir(constants.LOG_DIR)
        if not os.path.exists(constants.MODEL_DIR):
            os.makedirs(constants.MODEL_DIR)

        ################################
        # Get arguments and initialize trainer

        args = self.get_args()
        trainer = self.get_trainer(args)

        ################################
        # Prepare gpu

        use_gpu = 'gpu' in args and args.gpu >= 0
        if use_gpu:
            os.environ[constants.CUDA_VISIBLE_DEVICE] = str(args.gpu)

        ################################
        # Load external embedding model

        trainer.load_external_embedding_models()

        ################################
        # Load classifier model

        if args.model_path:
            trainer.load_model()
        else:
            trainer.init_hyperparameters()

        ################################
        # Setup feature extractor and initialize dic from external dictionary

        # trainer.init_feature_extractor(use_gpu)
        trainer.load_external_dictionary()
        trainer.load_subword_dictionary()

        ################################
        # Load dataset

        if args.execute_mode == 'train':
            trainer.load_training_and_validation_data()
        elif args.execute_mode == 'eval':
            trainer.load_test_data()
        elif args.execute_mode == 'decode':
            trainer.load_decode_data()

        ################################
        # Set up classifier

        if not trainer.classifier:
            trainer.init_model()
        else:
            trainer.update_model(train=args.execute_mode == 'train')

        if use_gpu:
            trainer.classifier.to_gpu()

        ################################
        # Run

        if args.execute_mode == 'train':
            trainer.setup_optimizer()
            trainer.setup_scheduler()
            trainer.setup_evaluator()
            trainer.run_train_mode()

        elif args.execute_mode == 'eval':
            trainer.setup_evaluator()
            trainer.run_eval_mode()

        elif args.execute_mode == 'decode':
            trainer.run_decode_mode()

        ################################
        # Terminate

        trainer.close()
