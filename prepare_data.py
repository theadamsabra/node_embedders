import argparse
from torch_geometric.datasets import Flickr


def download_flickr(save_path_flickr):
    Flickr(save_path_flickr)

def download_youtube(save_path_youtube):
    pass

def download_blog_cataloge(save_path_blog_cataloge):
    pass

if __name__ == "__main__":
    '''
    Script to automatically download data for you.
    '''
    parser = argparse.ArgumentParser(
        prog='Data Preparation'
    )
    # path to save flickr dataset:
    parser.add_argument('-f', '--flickr', type=str)
    # path to save youtube dataset:
    parser.add_argument('-y', '--youtube', type=str)
    # path to save blog cataloge dataset:
    parser.add_argument('-b', '--blogcataloge', type=str)

    parser.parse_args()

    # download flickr:
    download_flickr(parser.flickr) 
    # TODO: download youtube:
    download_youtube(parser.youtube)
    # TODO: download blog cataloge:
    download_blog_cataloge(parser.blogcataloge)