import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser()

        self._parser.add_argument('--dataset_path', type=str, default='dataset', help='Dataset file path.')
        self._parser.add_argument('--class_choice', type=str, default='Rocket', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
        self._parser.add_argument('--batch_size', type=int, default=32, help='Integer value for batch size.')
        self._parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')

        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--epochs', type=int, default=2001, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')

        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3,  64,  128, 256, 512, 512], nargs='+', help='Features for discriminator.')

    def parser(self):
        return self._parser