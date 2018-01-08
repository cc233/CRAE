import tensorflow as tf
import os
import doc2vec
class deepMF_with_text(object):

    def __init__(self,
                 user_num,
                 item_num,
                 latent_dim,
                 text_latent_dim,                 
                 batch_size,
                 learning_rate,
                 doc_index,
                 doc_index_reverse,
                 doc_mask,
                 doc_mask_bool,
                 word_vec,
                 optimizer='adam',
                 dtype=tf.float32,
                 scope='deepMF_with_text'):
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.text_latent_dim=text_latent_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.dtype = dtype
        self.doc_index=doc_index
        self.doc_index_reverse=doc_index_reverse
        self.doc_mask=doc_mask
        self.doc_mask_bool=doc_mask_bool
        self.word_vec=word_vec
        self.word_num=word_vec.shape[0]
        self.word_num+=1
        self.test_shape=[]
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=self.dtype, name='learning_rate')
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

            self.build_graph()
            self.saver = tf.train.Saver(max_to_keep=10)


    def build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_struct()
        self._create_square_loss()
        self._create_optimizer()


    def _create_placeholder(self):
        self.user_inputs=tf.placeholder(tf.int32,[None],name='user_inputs')
        self.pos_items=tf.placeholder(tf.int32,[None],name='positive_items')
        self.neg_items=tf.placeholder(tf.int32,[None,None],name='negative_items')

    def _create_embedding(self):
        self.user_embeddings = tf.Variable(tf.random_normal([self.user_num, self.latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='user_embeddings')
        self.item_embeddings = tf.Variable(tf.random_normal([self.item_num, self.latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='item_embeddings')
        self.doc_index=tf.Variable(self.doc_index,dtype=tf.int32,trainable=False,name='document_index')
        self.doc_index_reverse=tf.Variable(self.doc_index_reverse,dtype=tf.int32,trainable=False,name='document_index_reverse')
        self.doc_mask=tf.Variable(self.doc_mask,dtype=tf.int32,trainable=False,name='document_mask')
        self.doc_mask_bool=tf.Variable(self.doc_mask_bool,dtype=self.dtype,trainable=False,name='document_mask_bool')
        self.word_vec=tf.Variable(self.word_vec,dtype=tf.float32,trainable=True,name='word_vector')
        self.word_prediction_W=tf.Variable(tf.random_normal([self.word_num, self.text_latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='word_prediction_W')
        self.word_prediction_b=tf.Variable(tf.random_normal([self.word_num], stddev=0.01), dtype=self.dtype,trainable=True, name='word_prediction_b')

    def _create_struct(self):
        pos_doc_index=tf.nn.embedding_lookup(self.doc_index,self.pos_items,name='pos_doc_index')
        pos_doc_index_reverse=tf.nn.embedding_lookup(self.doc_index_reverse,self.pos_items,name='pos_doc_index_reverse')
        pos_doc_mask=tf.nn.embedding_lookup(self.doc_mask,self.pos_items,name='pos_doc_mask')
        pos_doc_mask_bool=tf.nn.embedding_lookup(self.doc_mask_bool,self.pos_items,name='pos_doc_mask_bool')
        
        neg_items_shape=tf.shape(self.neg_items)
        neg_items_reshape=tf.reshape(self.neg_items,[neg_items_shape[0]*neg_items_shape[1]],name='neg_items_reshape')
        neg_doc_index=tf.nn.embedding_lookup(self.doc_index,neg_items_reshape,name='neg_doc_index')
        neg_doc_index_reverse=tf.nn.embedding_lookup(self.doc_index_reverse,neg_items_reshape,name='neg_doc_index_reverse')
        neg_doc_mask=tf.nn.embedding_lookup(self.doc_mask,neg_items_reshape,name='neg_doc_mask')
        neg_doc_mask_bool=tf.nn.embedding_lookup(self.doc_mask_bool,neg_items_reshape,name='neg_doc_mask_bool')

        #get document vector
        pos_items_doc_vec,self.pos_word_prediction_loss,_=doc2vec.calc_doc_vec(
            doc_index=pos_doc_index,
            doc_index_reverse=pos_doc_index_reverse,
            mask=pos_doc_mask,
            mask_bool=pos_doc_mask_bool,
            word_vec=self.word_vec,
            depth=self.text_latent_dim,
            word_prediction_W=self.word_prediction_W,
            word_prediction_b=self.word_prediction_b,
            word_num=self.word_num,
            test_shape=self.test_shape
            )
        neg_items_doc_vec,self.neg_word_prediction_loss,self.test_shape=doc2vec.calc_doc_vec(
            doc_index=neg_doc_index,
            doc_index_reverse=neg_doc_index_reverse,
            mask=neg_doc_mask,
            mask_bool=neg_doc_mask_bool,
            word_vec=self.word_vec,
            depth=self.text_latent_dim,
            word_prediction_W=self.word_prediction_W,
            word_prediction_b=self.word_prediction_b,
            word_num=self.word_num,
            test_shape=self.test_shape
            )

        self.user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_inputs, name='users_embed')
        self.pos_items_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items, name='pos_items_embed')
        neg_items_embed=tf.nn.embedding_lookup(self.item_embeddings,neg_items_reshape,name='neg_items_embed')
        
        self.pos_items_embed=self.pos_items_embed+pos_items_doc_vec
        neg_items_embed=neg_items_embed+neg_items_doc_vec
        self.neg_items_embed=tf.reshape(neg_items_embed,[neg_items_shape[0],neg_items_shape[1],-1])

    def _create_square_loss(self):
        self.pos_scores=tf.reduce_sum(self.user_embed*self.pos_items_embed,axis=1)
        self.neg_scores=tf.reduce_sum(tf.expand_dims(self.user_embed,axis=1)*self.neg_items_embed,axis=2)
        pos_labels=tf.ones(tf.shape(self.pos_scores),tf.float32)
        neg_labels=tf.zeros(tf.shape(self.neg_scores),tf.float32)
        #self.pos_scores=self.pos_scores1+self.pos_scores
        #self.neg_scores=self.neg_scores1+self.neg_scores
        pos_loss=tf.losses.mean_squared_error(labels=pos_labels,predictions=self.pos_scores)
        neg_loss=tf.losses.mean_squared_error(labels=neg_labels,predictions=self.neg_scores)
        self.loss= (pos_loss+neg_loss)/2

        #word_prediction_loss
        word_prediction_loss=self.pos_word_prediction_loss+self.neg_word_prediction_loss
        #self.test_shape.append(word_prediction_loss)
        self.loss+=word_prediction_loss

    def _create_optimizer(self):
        #params = tf.trainable_variables()
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.updates = optimizer.minimize(self.loss, self.global_step)

        # gradients = tf.gradients(self.loss, params)
        # clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # self.updates = optimizer.apply_gradients(
        #     zip(clipped_gradients, params),
        #     global_step=self.global_step
        # )

    def step(self,session,user_inputs,pos_items,neg_items,training):
        input_feed={}
        input_feed[self.user_inputs.name]=user_inputs
        input_feed[self.pos_items.name]=pos_items
        input_feed[self.neg_items.name]=neg_items
        if training:
          output_feed=[self.loss,self.test_shape,self.updates]
        else:
          output_feed=[self.pos_scores]
        #testing:
        #output_feed=[self.test_shape]
        outputs=session.run(output_feed,input_feed)
        if len(outputs)==1:
            outputs.append(0)
        return outputs[0],outputs[1]