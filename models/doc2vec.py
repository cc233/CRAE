import tensorflow as tf
def doc2vec(word_embed,mask,mask_bool,depth,doc_index_reverse,word_num,word_prediction_W,word_prediction_b,test_shape):
    #word_embed:[batch_size,max_word,depth]
    #mask:[batch_size]
    #depth:depth
    #doc_index_reverse:[batch_size,max_word]
    shape=tf.shape(word_embed,name='word_embed_shape')
    word_embed.set_shape([None,None,depth])
    cell_encode=tf.contrib.rnn.GRUCell(num_units=depth)
    outputs_encode,last_state_encode=tf.nn.dynamic_rnn(cell=cell_encode,inputs=word_embed,sequence_length=mask,dtype=tf.float32)
    #outputs_encode:[batch_size,max_word,depth]
    #last_state_encode:[batch_size,depth]
    #GRU
    cell_decode=tf.contrib.rnn.GRUCell(num_units=depth)
    zeros=tf.zeros((shape[0],shape[1],shape[2]))
    zeros.set_shape([None,None,depth])
    outputs_decode,last_state_decode=tf.nn.dynamic_rnn(cell=cell_decode,inputs=zeros,sequence_length=mask,dtype=tf.float32)
    #outputs:[batch_size,max_word,depth]
    outputs=tf.concat([outputs_encode,outputs_decode],axis=1)
    doc_vec=tf.reduce_mean(outputs,axis=1)

    #loss
    rand=tf.random_uniform((shape[0]*shape[1],1),minval=0,maxval=1)
    rand=rand<0.8
    rand=tf.cast(rand,dtype=tf.int32)
    blank=tf.ones((shape[0]*shape[1],1),dtype=tf.int32)*(word_num-1)
    doc_index_reverse_flat=tf.reshape(doc_index_reverse,[shape[0]*shape[1],1],name='doc_index_reverse_flat')
    labels=rand*doc_index_reverse_flat+(1-rand)*blank
    test_shape.append(tf.shape(labels))
    losses=tf.nn.sampled_softmax_loss(
        weights=word_prediction_W,
        biases=word_prediction_b,
        labels=labels,
        inputs=tf.reshape(outputs_decode,[shape[0]*shape[1],shape[2]]),
        num_sampled=1000,
        num_classes=word_num
        )
    losses=tf.reshape(losses,[shape[0],shape[1]],name='loss_reshape')
    word_prediction_loss=tf.reduce_sum((losses*mask_bool)/tf.reduce_sum(mask_bool),name='word_prediction_loss')
    return doc_vec,word_prediction_loss,test_shape

def calc_doc_vec(doc_index,doc_index_reverse,mask,mask_bool,word_vec,depth,
                 word_prediction_W,word_prediction_b,word_num,test_shape):
    word_embed= tf.nn.embedding_lookup(word_vec, doc_index, name='word_embed')
    doc_vec,word_prediction_loss,test_shape=doc2vec(word_embed,mask,mask_bool,depth,doc_index_reverse,word_num,word_prediction_W,word_prediction_b,test_shape)
    return doc_vec,word_prediction_loss,test_shape
