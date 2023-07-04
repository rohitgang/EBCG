# This contains all the code required for feature generation and algorithms for the paper

def generate_feature_vectors(sentence, bucket):
    """
        The features for each pair p_ij = (S_i, ei_j) are (phi_S_i, phi_ei_j, phi_st_i).
            (1) phi_S_i : a vector where each elements shows the presence of a word from the sentence in the vocabulary
            (2) phi_ei_j : 
            (3) phi_st_i : a vector where each element denotes if the string argument is matched with a word in the sentence

        Input : 
            sentence : the commentary sentence
            bucket : a series of events that are assigned to the commentary sentence.
        Output :
            (1) sentence
            (2) a vector of tuples of event and feature vector
    """

    #---------------------------------- phi_S_i ----------------------------------#
    ## TODO
    # Initialize it with domain-specific high frequency words, excluding words 
    # that can appear in the string arguments for example player names.
    # Vocabulary can be a list of words
    vocabulary = None 
    for event in bucket:
        phi_S_i = [0 for i in range(len(sentence.split()))] # a binary vector, 0 means not present, 1 means present
        for i,word in enumerate(sentence):
            if word in vocabulary:
                phi_S_i[i] = 1

        # TODO : Implementing phi_ei_j and phi_st_i

    
    return None
        


    
