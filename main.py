#-*-encoding=utf8-*-
from flask import jsonify
from flask import Flask
from flask import request
import json
import platform
import codecs
import logging
import itertools
from collections import OrderedDict
import os
import sys
from gevent import monkey
monkey.patch_all()
import tensorflow as tf
import numpy as np
from MODEL.NER.NER_TENSORFLOW.model import Model
from MODEL.NER.NER_TENSORFLOW.utils import get_logger,load_config,create_model
from MODEL.NER.NER_TENSORFLOW.utils import make_path
from MODEL.NER.NER_TENSORFLOW.utils import print_config, save_config, load_config, test_ner
from MODEL.NER.NER_TENSORFLOW.data_utils import load_word2vec, create_input, input_from_line, BatchManager
currentPath=os.getcwd()
sys.path.append(currentPath)
root_path=os.getcwd()

global pyversion
if sys.version>'3':
    pyversion='three'
else:
    pyversion='two'
if pyversion=='three':
    import pickle
else :
    import cPickle,pickle
root_path=os.getcwd()+os.sep
print("root_path:",root_path)
flags = tf.app.flags
flags.DEFINE_boolean("train",       False,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iob",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       True,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "example.test_text.txt"),   "Path for test_text.txt data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    print("char_to_id:",char_to_id)
    print("tag_to_id:",tag_to_id)
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


with open(FLAGS.map_file, "rb") as f:
    if pyversion=='three':    
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    else:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f,protocol=2)
        # make path for store log and model if not exist
make_path(FLAGS)
if os.path.isfile(FLAGS.config_file):
    print('flag:',FLAGS.config_file)
    config = load_config(FLAGS.config_file)
else:
    print('char_to_id:',type(char_to_id))
    print('tag_to_id:',type(tag_to_id))
    config = config_model(char_to_id, tag_to_id)
    save_config(config, FLAGS.config_file)
make_path(FLAGS)
app = Flask(__name__)
log_path = os.path.join("log", FLAGS.log_file)
logger = get_logger(log_path)


'''
如上图报错，说gpu没有支持的kernel。原因是应为在tensorflow中，定义在图中的op，有的只能再cup中运行，gpu中不支持。
解决方法就是让op自动识别，让它选择在合适的地方运行即可。如果op中有标识的话，在运行的时候指定在cup上执行，如果无法区分的话，
可以试试在sess.run的时候加入allow_soft_placement=True。例如：sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
'''
tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
print(111222333)
#print("char_to_id::::: %s " % char_to_id)
#print("char_to_id::::: %s " % tag_to_id)


@app.route('/ner', methods=['POST','GET'])
def get_text_input():
    #return "connect successfully"
    logging.info("connect successfully")
    if request.method == "POST":
        # versionCode = request.form.get("version")
        text = request.form.get("inputStr")
    else:
        # versionCode = request.args.get("version")
        text = request.args.get("inputStr")
    #return text
    #text="乙肝倒计时"
    if text:
        per, loc, org, tm = model.evaluate_line(sess, input_from_line(text, char_to_id), id_to_tag)
        ner_result = dict()
        print('per, loc, org,tm:', per, loc, org, tm)
        ner_result['per'] = per
        ner_result['loc'] = loc
        ner_result['org'] = org
        ner_result['tm'] = tm
        return json.dumps(ner_result)
if __name__ == "__main__":   
    # app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1',port=8097)
   # r=requests.post('http://192.168.5.40:5001/Neo4jAPI/P_relation', data={'patient_id':'10502005'})

