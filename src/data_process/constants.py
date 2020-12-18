import os


class Const:
    index_file_name = 'imagelist.txt'
    img_file_dir = 'image'

    PNET = 'PNET'
    RNET = 'RNET'
    ONET = 'ONET'

    # root_path代表项目的根路径
    root_path = os.path.abspath(os.path.dirname(__file__)).split('MyMtcnn')[0]
    root_path = os.path.join(root_path, 'MyMtcnn')
    print(root_path)

    pnet_positive_data_dir = 'positive_12'
    pnet_negative_data_dir = 'negative_12'
    pnet_part_data_dir = 'part_12'
    pnet_tfrecord = os.path.join(root_path, 'pnet_tfrecord')

    rnet_positive_data_dir = 'positive_24'
    rnet_negative_data_dir = 'negative_24'
    rnet_part_data_dir = 'part_24'
    rnet_tfrecord = os.path.join(root_path, 'rnet_tfrecord')

    onet_positive_data_dir = 'positive_48'
    onet_negative_data_dir = 'negative_48'
    onet_part_data_dir = 'part_48'
    onet_tfrecord = os.path.join(root_path, 'onet_tfrecord')


    LABEL_N = -1
    LABEL_POSI = 1
    LABEL_PART = 0



