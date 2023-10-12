#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

#run on cpu mode
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor apf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
