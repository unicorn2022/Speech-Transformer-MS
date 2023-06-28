import mindspore
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, save_checkpoint



if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.ckpt'
    checkpoint = load_checkpoint(checkpoint)
    model = Model()  # Replace Model() with your MindSpore model definition
    model.load_state_dict(checkpoint['model'])

    save_checkpoint(model.state_dict(), 'speech-transformer-cn.ckpt')

