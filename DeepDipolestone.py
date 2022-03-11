#!/usr/bin/env python3

import os,sys
import numpy as np
from deepmd.DeepEval import DeepTensor

class DeepDipolestone (DeepTensor) :
    def __init__(self,
                 model_file) :
        DeepTensor.__init__(self, model_file, 'dipolestone', 3)


