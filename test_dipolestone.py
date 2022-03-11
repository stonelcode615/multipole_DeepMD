import dpdata,os,sys,json,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data

from deepmd.RunOptions import RunOptions
from deepmd.DataSystem import DataSystem
from deepmd.DescrptLocFrame import DescrptLocFrame
from deepmd.Fitting import DipolestoneFitting
from deepmd.Model import DipolestoneModel
from deepmd.common import j_must_have, j_must_have_d, j_have

global_ener_float_precision = tf.float64
global_tf_float_precision = tf.float64
global_np_float_precision = np.float64

class TestModel(unittest.TestCase):
    def setUp(self) :
        gen_data()

    def test_model(self):
        jfile = 'dipolestone.json'
        with open(jfile) as fp:
            jdata = json.load (fp)
        run_opt = RunOptions(None)
        systems = j_must_have(jdata, 'systems')
        set_pfx = j_must_have(jdata, 'set_prefix')
        batch_size = j_must_have(jdata, 'batch_size')
        test_size = j_must_have(jdata, 'numb_test')
        batch_size = 1
        test_size = 1
        stop_batch = j_must_have(jdata, 'stop_batch')
        rcut = j_must_have (jdata['model']['descriptor'], 'rcut')

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt = None)

        test_data = data.get_test ()
        numb_test = 1

        descrpt = DescrptLocFrame(jdata['model']['descriptor'])
        fitting = DipolestoneFitting(jdata['model']['fitting_net'], descrpt)
        model = DipolestoneModel(jdata['model'], descrpt, fitting)

        input_data = {'coord' : [test_data['coord']],
                      'box': [test_data['box']],
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']],
                      'fparam': [test_data['fparam']],
        }
        model._compute_dstats(input_data)



        t_prop_c           = tf.placeholder(tf.float32, [5],    name='t_prop_c')
        t_energy           = tf.placeholder(global_ener_float_precision, [None], name='t_energy')
        t_force            = tf.placeholder(global_tf_float_precision, [None], name='t_force')
        t_virial           = tf.placeholder(global_tf_float_precision, [None], name='t_virial')
        t_atom_ener        = tf.placeholder(global_tf_float_precision, [None], name='t_atom_ener')
        t_coord            = tf.placeholder(global_tf_float_precision, [None], name='i_coord')
        t_type             = tf.placeholder(tf.int32,   [None], name='i_type')
        t_natoms           = tf.placeholder(tf.int32,   [model.ntypes+2], name='i_natoms')
        t_box              = tf.placeholder(global_tf_float_precision, [None, 9], name='i_box')
        t_mesh             = tf.placeholder(tf.int32,   [None], name='i_mesh')
        is_training        = tf.placeholder(tf.bool)
        t_fparam = None

        model_pred \
            = model.build (t_coord,
                           t_type,
                           t_natoms,
                           t_box,
                           t_mesh,
                           t_fparam,
                           suffix = "dipolestone",
                           reuse = False)
        dipolestone = model_pred['dipolestone']

        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_coord:         np.reshape(test_data['coord']    [:numb_test, :], [-1]),
                          t_box:           test_data['box']                 [:numb_test, :],
                          t_type:          np.reshape(test_data['type']     [:numb_test, :], [-1]),
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False}

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [p] = sess.run([dipolestone], feed_dict = feed_dict_test)

        p = p.reshape([-1])
        refp = [-9.105016838228578990e-01,7.196284362034099935e-01,-9.548516928185298014e-02,2.764615027095288724e+00,2.661319598995644520e-01,7.579512949131941846e-02]

        places = 6
        for ii in range(p.size) :
            self.assertAlmostEqual(p[ii], refp[ii], places = places)





