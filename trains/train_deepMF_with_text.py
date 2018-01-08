import tensorflow as tf
import numpy as np
from time import time
import math
import os,sys
import heapq
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)
from models.deepMF_with_text import deepMF_with_text
from utils import evaluation
from data.data import Dataset

#citeulike-a
tf.app.flags.DEFINE_string('rating_matrix_path', '/home/chenchen/data/ctrsr_datasets/citeulike-a/users.dat', 'rating matrix data path')
tf.app.flags.DEFINE_string('text_path', '/home/chenchen/data/ctrsr_datasets/citeulike-a/raw-data.csv', 'text data path')
tf.app.flags.DEFINE_string('word2vec_model_path', '/home/chenchen/recommendation/data/citeulike-a_word2vec_model_32', 'word2vec model data path')
tf.app.flags.DEFINE_string('tag_item_path', '/home/chenchen/data/ctrsr_datasets/citeulike-a/pre_tag_item.dat', 'tag item path')
tf.app.flags.DEFINE_string('data_set', 'citeulike-a', 'data set')

#citeulike-t
# tf.app.flags.DEFINE_string('rating_matrix_path', '/home/chenchen/data/ctrsr_datasets/citeulike-t/users.dat', 'rating matrix data path')
# tf.app.flags.DEFINE_string('text_path', '/home/chenchen/data/ctrsr_datasets/citeulike-t/rawtext.dat', 'text data path')
# tf.app.flags.DEFINE_string('word2vec_model_path', '/home/chenchen/recommendation/data/citeulike-t_word2vec_model_32', 'word2vec model data path')
# tf.app.flags.DEFINE_string('tag_item_path', '/home/chenchen/data/ctrsr_datasets/citeulike-t/tag-item.dat', 'tag item path')
# tf.app.flags.DEFINE_string('data_set', 'citeulike-t', 'data set')

tf.app.flags.DEFINE_string('log_path', '../ckpt/deepMF_with_text_log/', 'model save path')
tf.app.flags.DEFINE_integer('epochs', 500, 'number of epochs')
tf.app.flags.DEFINE_integer('latent_dim', 32, 'embedding size')
tf.app.flags.DEFINE_integer('text_latent_dim', 32, 'text embedding size')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.app.flags.DEFINE_integer('num_neg', 8, 'number of negative instances to pair with a positive instance')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'specify an optimizer: adagrad, adam, rmsprop, sgd')
tf.app.flags.DEFINE_integer('verbose', 1000, 'show performance per X iterations')
tf.app.flags.DEFINE_integer('test_itr',3000,'test performance per X iterations')
tf.app.flags.DEFINE_integer('save',2000,'save every X iterations')
tf.app.flags.DEFINE_string('train_dir', '../ckpt/deepMF_with_text/', 'model save path')
FLAGS = tf.app.flags.FLAGS


def create_model(user_num, item_num, ckpt_path, optimizer, session,doc_index,doc_index_reverse,doc_mask,doc_mask_bool,word_vec):
    model = deepMF_with_text(
        user_num=user_num,
        item_num=item_num,
        latent_dim=FLAGS.latent_dim,
        text_latent_dim=FLAGS.text_latent_dim,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        doc_index=doc_index,
        doc_index_reverse=doc_index_reverse,
        doc_mask=doc_mask,
        doc_mask_bool=doc_mask_bool,
        word_vec=word_vec,
        optimizer=optimizer
    )

    #ckpt = tf.train.get_checkpoint_state(ckpt_path)
    #if ckpt and ckpt.model_checkpoint_path:
    #    print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
    #    model.saver.restore(session, ckpt.model_checkpoint_path)
    #else:
    print('Creating model with fresh parameters.')
    session.run(tf.global_variables_initializer())

    return model

def evaluate(model,sess,dataset):
    num_user_test=500
    #top_k=50
    user_num=model.user_num

    #get random users
    perm=np.arange(user_num)
    np.random.shuffle(perm)
    if(num_user_test>user_num):
        num_user_test=user_num
    perm=perm[0:num_user_test]
    temp=np.zeros([1,1])
    auc_all=0
    precision50_all=0
    precision100_all=0
    precision150_all=0
    precision200_all=0
    precision250_all=0
    precision300_all=0
    recall50_all=0
    recall100_all=0
    recall150_all=0
    recall200_all=0
    recall250_all=0
    recall300_all=0
    average_precision_all=0
    ndcg_all=0
    users_with_zero_item=0
    for i in xrange(num_user_test):
        user_id=perm[i]
        BPR_test_items,labels=dataset.get_BPR_test(user_id)
        users=np.zeros(BPR_test_items.shape)
        users[:]=user_id
        temp=np.zeros(BPR_test_items.shape)
        temp=np.expand_dims(temp,axis=1)
        pos_scores,_=model.step(sess,users,BPR_test_items,temp,False)
        #auc=sklearn.metrics.roc_auc_score(y_true=labels,y_score=pos_scores)
        #pos_scores=pos_scores.reshape([-1])
        #print pos_scores.shape
        if np.sum(labels)==0:
            users_with_zero_item+=1
            continue

        auc=evaluation.cal_auc(labels,pos_scores)
        precision50=evaluation.cal_precision(labels,pos_scores,top_k=50)
        precision100=evaluation.cal_precision(labels,pos_scores,top_k=100)
        precision150=evaluation.cal_precision(labels,pos_scores,top_k=150)
        precision200=evaluation.cal_precision(labels,pos_scores,top_k=200)
        precision250=evaluation.cal_precision(labels,pos_scores,top_k=250)
        precision300=evaluation.cal_precision(labels,pos_scores,top_k=300)
        recall50=evaluation.cal_recall(labels,pos_scores,top_k=50)
        recall100=evaluation.cal_recall(labels,pos_scores,top_k=100)
        recall150=evaluation.cal_recall(labels,pos_scores,top_k=150)
        recall200=evaluation.cal_recall(labels,pos_scores,top_k=200)
        recall250=evaluation.cal_recall(labels,pos_scores,top_k=250)
        recall300=evaluation.cal_recall(labels,pos_scores,top_k=300)
        average_precision=evaluation.cal_map(labels,pos_scores)
        ndcg=evaluation.cal_NDCG(pos_scores)
        precision50_all+=precision50
        precision100_all+=precision100
        precision150_all+=precision150
        precision200_all+=precision200
        precision250_all+=precision250
        precision300_all+=precision300
        recall50_all+=recall50
        recall100_all+=recall100
        recall150_all+=recall150
        recall200_all+=recall200
        recall250_all+=recall250
        recall300_all+=recall300
        average_precision_all+=average_precision
        ndcg_all+=ndcg
        #print 'Recall:'+str(recall)
        auc_all+=auc
    auc_all=auc_all/(num_user_test-users_with_zero_item)
    precision50_all=precision50_all/(num_user_test-users_with_zero_item)
    precision100_all=precision100_all/(num_user_test-users_with_zero_item)
    precision150_all=precision150_all/(num_user_test-users_with_zero_item)
    precision200_all=precision200_all/(num_user_test-users_with_zero_item)
    precision250_all=precision250_all/(num_user_test-users_with_zero_item)
    precision300_all=precision300_all/(num_user_test-users_with_zero_item)
    recall50_all=recall50_all/(num_user_test-users_with_zero_item)
    recall100_all=recall100_all/(num_user_test-users_with_zero_item)
    recall150_all=recall150_all/(num_user_test-users_with_zero_item)
    recall200_all=recall200_all/(num_user_test-users_with_zero_item)
    recall250_all=recall250_all/(num_user_test-users_with_zero_item)
    recall300_all=recall300_all/(num_user_test-users_with_zero_item)
    map=average_precision_all/(num_user_test-users_with_zero_item)
    ndcg_all=ndcg_all/(num_user_test-users_with_zero_item)
    #print (num_user_test-users_with_zero_item)
    return auc_all,precision50_all,precision100_all,precision150_all,precision200_all,precision250_all,precision300_all,map,ndcg_all,recall50_all,recall100_all,recall150_all,recall200_all,recall250_all,recall300_all

def train():
    print 'Learning rate:'+str(FLAGS.learning_rate)
    print 'Num neg:'+str(FLAGS.num_neg)
    print 'Latent dim:'+str(FLAGS.latent_dim)
    print 'Batch size:'+str(FLAGS.batch_size)
    ckpt_path = FLAGS.train_dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    #print ckpt_path
    #load data
    dataset=Dataset(rating_matrix_path=FLAGS.rating_matrix_path,
        num_negatives=FLAGS.num_neg,
        data_set=FLAGS.data_set,
        text_path=FLAGS.text_path,
        word2vec_model_path=FLAGS.word2vec_model_path
        )
    

    user_num, item_num = dataset.get_user_item_num()
    print("Load data done. #user=%d, #item=%d, "
          % (user_num, item_num ))
    train_num=dataset.get_train_num()
    print('train_num=%d'%(train_num))
    doc_index,doc_index_reverse,doc_mask,doc_mask_bool,word_vec=dataset.get_doc()
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(user_num, item_num, ckpt_path, FLAGS.optimizer, sess,doc_index,doc_index_reverse,doc_mask,doc_mask_bool,word_vec)
        #writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        iterations=train_num*FLAGS.epochs/FLAGS.batch_size+1
        print 'Iteration:'+str(iterations)
        #training
        loss_sum=0
        itr_print=FLAGS.verbose
        itr_save=FLAGS.save
        itr_test=FLAGS.test_itr
        highest_auc=0
        highest_precision_50=0
        highest_precision_100=0
        highest_precision_150=0
        highest_precision_200=0
        highest_precision_250=0
        highest_precision_300=0
        highest_recall_50=0
        highest_recall_100=0
        highest_recall_150=0
        highest_recall_200=0
        highest_recall_250=0
        highest_recall_300=0
        highest_map=0
        highest_ndcg=0
        for itr in xrange(iterations):
            users, pos_items, neg_items = dataset.next_BPR_batch(FLAGS.batch_size)
            loss,test = model.step(sess, users, pos_items, neg_items,True)
            #print loss
            #testing
            #test=model.step(sess, users, pos_items, neg_items,True)
            #print test
            loss_sum+=loss
            #if((itr%itr_save==0 and itr!=0) or itr==iterations-1):
            #    model.saver.save(sess,ckpt_path+'train',model.global_step)
            #print loss
            if(itr%itr_print==0 and itr!=0):
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Average Loss:'+str(loss_sum/itr_print)
                loss_sum=0
            #print loss
            if(itr==(iterations-1) and itr%itr_print!=0):
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Average Loss:'+str(loss_sum/(itr%itr_print))
                loss_sum=0
            #test
            if(itr%itr_test==0 or itr==iterations-1):
                auc,precision_50,precision_100,precision_150,precision_200,precision_250,precision_300,map,ndcg,recall_50,recall_100,recall_150,recall_200,recall_250,recall_300=evaluate(model,sess,dataset)
                if auc>highest_auc:
                    highest_auc=auc
                if precision_50>highest_precision_50:
                    highest_precision_50=precision_50
                if precision_100>highest_precision_100:
                    highest_precision_100=precision_100
                if precision_150>highest_precision_150:
                    highest_precision_150=precision_150
                if precision_200>highest_precision_200:
                    highest_precision_200=precision_200
                if precision_250>highest_precision_250:
                    highest_precision_250=precision_250
                if precision_300>highest_precision_300:
                    highest_precision_300=precision_300
                if map>highest_map:
                    highest_map=map
                if ndcg>highest_ndcg:
                    highest_ndcg=ndcg
                if recall_50>highest_recall_50:
                    highest_recall_50=recall_50
                if recall_100>highest_recall_100:
                    highest_recall_100=recall_100
                if recall_150>highest_recall_150:
                    highest_recall_150=recall_150
                if recall_200>highest_recall_200:
                    highest_recall_200=recall_200
                if recall_250>highest_recall_250:
                    highest_recall_250=recall_250
                if recall_300>highest_recall_300:
                    highest_recall_300=recall_300
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' MAP:'+str(map)+' Highest MAP:'+str(highest_map)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Precision 50:'+str(precision_50)+' Highest Precision 50:'+str(highest_precision_50)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Precision 100:'+str(precision_100)+' Highest Precision 100:'+str(highest_precision_100)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Precision 150:'+str(precision_150)+' Highest Precision 150:'+str(highest_precision_150)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Precision 200:'+str(precision_200)+' Highest Precision 200:'+str(highest_precision_200)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Precision 250:'+str(precision_250)+' Highest Precision 250:'+str(highest_precision_250)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Precision 300:'+str(precision_300)+' Highest Precision 300:'+str(highest_precision_300)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' NDCG:'+str(ndcg)+' Highest NDCG:'+str(highest_ndcg)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' AUC:'+str(auc)+' Highest AUC:'+str(highest_auc)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Recall 50:'+str(recall_50)+' Highest Recall 50:'+str(highest_recall_50)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Recall 100:'+str(recall_100)+' Highest Recall 100:'+str(highest_recall_100)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Recall 150:'+str(recall_150)+' Highest Recall 150:'+str(highest_recall_150)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Recall 200:'+str(recall_200)+' Highest Recall 200:'+str(highest_recall_200)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Recall 250:'+str(recall_250)+' Highest Recall 250:'+str(highest_recall_250)
                print 'Epoch:'+str(dataset.get_epoch())+' Iteration:'+str(itr)+' Recall 300:'+str(recall_300)+' Highest Recall 300:'+str(highest_recall_300)
                sys.stdout.flush()
    #writer.close()
def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
