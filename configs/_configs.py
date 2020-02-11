class Args():
    # debug
    debug = True

    # source
    train_data = './data/train.csv'

    # trained model path
    bert_base_uncased = './trained_model/bert-base-uncased'
    trained_uncased_bert = '/trained_model/trained_uncased_bert'

    # model parameters
    hidden_dropout_prob = 0.5
    hidden_size = 10
    num_labels = 15
    
    # datasets cols
    toxicity_column = 'target'
    identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    aux_columns = ['target', 'severe_toxicity','obscene','identity_attack','insult','threat','sexual_explicit']

    # training parameters
    batch_size = 32
    max_sequence_length = 220
    lr = 2e-5
