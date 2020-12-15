import os
class Const:
    index_file_name = 'imagelist.txt'
    img_file_dir = 'image'

    # root_path代表项目的根路径
    root_path = os.path.abspath(os.path.dirname(__file__)).split('MyMtcnn')[0]
    root_path = os.path.join(root_path, 'MyMtcnn')
    print(root_path)

    pnet_positive_data_dir = 'positive_12'
    pnet_negative_data_dir = 'negative_12'
    pnet_part_data_dir = 'part_12'

    LABEL_N = -1
    LABEL_POSI = 1
    LABEL_PART = 0
