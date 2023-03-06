from common.models import SVS
from processor.processor import Processor


def main():
    svs = SVS(scene="./data_raw/Col_PNI2021chall_train_0016.svs")
    tumor_model, nerve_model, pni_model = (
        "./path_to_model",
        "./path_to_model",
        "./path_to_model",
    )
    output = Processor(tumor_model, nerve_model, pni_model).process(svs)


if __name__ == "__main__":
    main()
