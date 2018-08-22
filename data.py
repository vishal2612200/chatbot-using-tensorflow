import numpy as np 
import tensorflow as tf 
import re 
import time

##data processing
# importing the dataset
lines = open('movie_lines.txt',encoding = 'utf-8',errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding = 'utf-8',errors ='ignore').read().split('\n')

#creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))

#getting separately the questions and the answers
questions=[]
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

#doing the cleaning of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", " will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>+=~.?,]", "", text)
    return text

# cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#cleaning the answer
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

#creating a dictionary that maps each word to its number of occurances
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

#creating two dictionaries that map the questions words and the answers words to a unique integer
threshold = 20
questionwords2int = {}
word_number = 0
for word,count in word2count.item():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number += 1 
    

answerwords2int = {}
word_number = 0
for word,count in word2count.item():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1 

#adding the last token to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1
#creating the inverse dictionary of the answerawords2int dictionary
answersints2word = {w_i:w for w,w_i in answersints2word.items()}

#adding the end of string token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'


#translating all the questions and the answer into integers
#and replacing all the words that were filtered out by <OUT>
question_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    question_into_int.append(ints)

answer_into_int = []
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerawords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append (answerwords2int[word])
    answer_into_int.append(ints)

#sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 26):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(question_into_int[i[0]])




######part-2 -BUILDING THE SEQ2SEQ MODEL ######
#creating placeholder for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32,[None, None],name = 'input') 
    targets = tf.placeholder(tf.int32,[None, None],name = 'target')
    lr = tf.placeholder(tf.float32,name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs,targets,lr,keep_prob

#preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets


#creating the encoder RNN Layer
#rnn_input = model inputs
#rnn_size = no. of input tensor in layer
#keep_prob = control the dropout(deactivation) rate of neuron
# _, = this is used to return specified output as at the end this function will return two output but we have return only one output i.e encoder_state .that why we are use this symbol in front of him


def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = encoder_cell,
                                                      sequence_length = sequence_length,
                                                      inputs = rnn_input,
                                                      dtype = tf.float32) 
    return encoder_state

#decording the training set
#1 = no. of column

# tf.zeros = 3d input having three axis i.e batch _size,no. of column, size of decoder cell output
#attention_keys are keys that are compared with target state
#attention_value are value that are used to construct context vector

#attention_score_function is used to compute similarties between the keys and target state
# attention_construct_function is a function used to build the construction of attention state 
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size ):
    attention_states = tf.zeros([batch_size , 1, decoder_cell.output_size])
    attention_keys,attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,attention_values,attention_score_function,attention_construct_function,
                                                                              name = 'attn_dec_train')
    decoder_output,decoder_final_state, decoder_final_context_state = tf.contib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                           training_decoder_function,
                                                                                                           decoder_embedded_input,
                                                                                                           sequence_length,
                                                                                                           scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

#Decoding the test/validation set 
#num_words total number of word in question/answer
#decorder_final_state and decoder_final_context_state are basic for our information otherwise they are useless in our code
#decoder_embedding_matrix is similar to decoder_embedding_input 
def decode_text_set(encoder_state,decoder_cell,decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function, attention_construct_function  = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau',num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_construct_function,
                                                                              decoder_embedding_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")

    test_predications, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                 test_decoder_function,
                                                                                                                 scope = decoding_scope)
    return text_predications

#creating the decorder rnn
def decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size ):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stdev= 0.1)
        biases = tf.zeros_initializer()#zeros_initializer is tensorflow function
        output_function = lambda x : tf.contrib.layers.fully_connected(x,
                                                                       num_words,
                                                                       None,
                                                                       scope = decoding_scope,
                                                                       weights_initializer = weights,
                                                                       biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variable()
        test_predications = decode_test_set(encoder_state,
                                            decoder_cell,
                                            decoder_embeddings_matrix,
                                            word2int['<SOS>'],
                                            word2int['<EOS>'],
                                            sequence_length - 1,#to not include last token
                                            num_words,
                                            decoding_scope,
                                            output_function,
                                            keep_prob,
                                            batch_size)
        return training_predictions , test_predications 
#num_words are the total number of words in answer
#tensorflow function = random_uniform_initializer ( size of matrix, minvalue, maxvalue)#initializer that generates tensorswith a uniform distribution
#encoder_state is output of the encoder that become input of decoder
#creating the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length,answer_num_words,question_num_words, encode_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionwords2int ):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answer_num_words +1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn_layer(encoder_embedded_input, rnn_size, keep_prob, num_layers, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionwords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([question_num_words+1, decoder_embedding_size ], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets )
    training_predictions , test_predictions = decoder_rnn(decoder_embedded_input,
                                                          decoder_embeddings_matrix,
                                                          encoder_state,
                                                          questions_num_words,
                                                          sequence_length,
                                                          rnn_size,num_layers,
                                                          questionswords2int,
                                                          keep_prob,
                                                          batch_size)
    return training_predictions, test_predications
####part3 - training the seq2seq model ####
#setting the hyperparameters
epochs = 100 
batch_size = 128
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#defining a session
#reset_default_graph() is a function which reset tensorflow graph according to our data

tf.reset_default_graph()
session = tf.InteractiveSession()

#modal inputs
inputs, targets, lr, keep_prob = model_inputs()

#setting the sequence length 
#placeholder_with_default(input,shape,name),inputs =25 means that we are choosing only question that have ony 25 words
sequence_length = tf.placeholder_with_default(25 , None, name='sequence_length')

#getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

#getting the training and test prediction
training_predictions, test_predications = seq2seq_model(tf.reverse(inputs,[-1]),
                                                        targets,
                                                        keep_prob,
                                                        batch_size,
                                                        sequence_length,
                                                        len(answerswords2int),
                                                        len(questionswords2int),
                                                        encoding_embedding_size,
                                                        decoding_embedding_size,
                                                        rnn_size,
                                                        num_layers,
                                                        questionswords2int
                                                        )


#setting up the loss error, the optimizer and gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_Loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0],sequence_length]))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients =  optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable) for grad_tensor,grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

#padding the sequence with the <PAD> token
#Question : ['who', 'are', 'you']
#Answer : [<SOS>, 'I', 'am', 'a', 'bot' , '.', <EOS> ]

#after applying the padding
#Questions : ['who', 'are', 'you', <PAD>, <PAD>,<PAD>,<PAD>]
#Answer    : [<SOS>, 'I'  , 'am',  'a' , 'bot', '.' ,<EOS>]

def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])#this for loop will give the length of total number of words in sequence
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences] 

#spliting the data into batches of question and answer
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index:start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerawords2int))
        yield padded_questions_in_batch , padded_answers_in_batch

#splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)#this is done to seprate 15% data from the original data for testing purpose
training_questions = sorted_clean_questions[training_validation_split:] # this is done to use remaining 85% data for training purpose
validation_questions = sorted_clean_questions[:training_validation_split] # here we start taking the list from the starting because now we using the 85% data in previous one we spliting the 85% data 
validation_answers = sorted_clean_answers[:training_validation_split]

#Training 
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0 
list_validation_loss_error = []
early_stopping_check = 0 
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error, {inputs: padded_questions_in_batch,
                                                                                              targets:padded_answers_in_batch,
                                                                                              lr: learning_rate,
                                                                                              sequence_length : padded_answers_in_batch.shape[1],
                                                                                              keep_prob : keep_probability}])
        #_, means that first element we don't want to return
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                                                                       #here the probability is one as this is a case of validation and in validation probability is kept to be one to keep the neuron alive at every time

                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
           #here we applying learning rate decay
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


###testing the seq2seq model

#loaading the weights and running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saveer()
saver.restore(session, checkpoint)

#converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
    #here we use get because to filter out less frequent word to replace less frequent word with an token id

#setting up the chat 
while(True):
    question = input("You: ")
    if question == 'Godbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']]*(20 - len(question))
    #nn take only batch so we have to convert question into batch
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs:fake_batch, keep_prob:0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i' :
            token = 'I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
