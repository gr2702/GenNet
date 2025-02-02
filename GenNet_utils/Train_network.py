import os
import sys
import warnings
import shutil
import matplotlib
import datetime
warnings.filterwarnings('ignore')
matplotlib.use('agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import tensorflow.keras as K

tf.keras.backend.set_epsilon(0.0000001)
from GenNet_utils.Dataloader import *
from GenNet_utils.Utility_functions import *
from GenNet_utils.Create_network import *
from GenNet_utils.Create_plots import *


def weighted_binary_crossentropy(y_true, y_pred):
    # Adjusting true and predicted values to avoid numerical issues
    y_true = K.backend.clip(tf.cast(y_true, dtype=tf.float32), 0.0001, 1)
    y_pred = K.backend.clip(tf.cast(y_pred, dtype=tf.float32), 0.0001, 1)
    # Compute weighted binary crossentropy
    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)


def train_classification(args):
    SlURM_JOB_ID = get_SLURM_id()
    
    # Initialize variables for model and masks
    model = None
    masks = None
    
    # Setting up paths and training parameters from the input arguments
    datapath = args.path
    jobid = args.ID
    wpc = args.wpc
    lr_opt = args.learning_rate
    batch_size = args.batch_size
    #epochs = args.epochs
    epochs = 100
    l1_value = args.L1
    L1_act = args.L1_act
    problem_type = args.problem_type
    patience = args.patience
    one_hot=args.onehot

    # Determine genotype path
    if args.genotype_path == "undefined":
        genotype_path = datapath
    else:
        genotype_path = args.genotype_path

    # Enable mixed precision if specified
    if args.mixed_precision == True:
        use_mixed_precision()

    # Set multiprocessing based on the number of workers specified
    if args.workers > 1:
        multiprocessing = True
    else:
        multiprocessing = False

    # Check data integrity and format
    check_data(datapath=datapath, genotype_path=genotype_path, mode=problem_type)

    # Initialize global variables for class weights
    global weight_positive_class, weight_negative_class

    weight_positive_class = wpc
    weight_negative_class = 1

    # Set up the optimizer for the model
    optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)

    # Calculate dataset sizes for training, validation, and test sets
    train_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 1)
    val_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 2)
    test_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 3)
    num_covariates = pd.read_csv(datapath + "subjects.csv").filter(like='cov_').shape[1]
    val_size_train = val_size

    # Adjust epoch size if specified
    if args.epoch_size is None:
        args.epoch_size = train_size
    else:
        val_size_train = min(args.epoch_size // 2, val_size)
        print("Using each epoch", args.epoch_size,"randomly selected training examples")
        print("Validation set size used during training is also set to half the epoch_size")

    # Determine input size for the network
    inputsize = get_inputsize(genotype_path)

    folder, resultpath = get_paths(args)

    
    print("jobid =  " + str(jobid))
    print("folder = " + str(folder))
    print("resultpath = " + str(resultpath))
    print("weight_possitive_class", weight_positive_class)
    print("weight_negative_class", weight_negative_class)
    print("batchsize = " + str(batch_size))
    print("lr = " + str(lr_opt))
    print("L1 = " + str(l1_value))
    print("L1_act = " + str(L1_act))
    print("onehot = " + str(one_hot))
    print("model = " + str(args.network_name))

    

    if args.network_name == "lasso":
        print("lasso network")
        model, masks = lasso(inputsize=inputsize, l1_value=l1_value, L1_act =L1_act)
        
    elif args.network_name == "sparse_directed_gene_l1":
        print("sparse_directed_gene_l1 network")
        model, masks = sparse_directed_gene_l1(datapath=datapath, inputsize=inputsize, l1_value=l1_value, one_hot=one_hot)
    elif args.network_name == "gene_network_multiple_filters":
        print("gene_network_multiple_filters network")
        model, masks = gene_network_multiple_filters(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act=L1_act, 
                                                   regression=False,num_covariates=num_covariates,
                                                   filters=args.filters, one_hot=one_hot)
    elif args.network_name == "gene_network_snp_gene_filters":
        print("gene_network_snp_gene_filters network")
        model, masks = gene_network_snp_gene_filters(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act,
                                                   regression=False, num_covariates=num_covariates,
                                                   filters=args.filters, one_hot=one_hot)
    else:
        if os.path.exists(datapath + "/topology.csv"):
            model, masks = create_network_from_csv(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act, num_covariates=num_covariates, 
                                                   one_hot=one_hot)
        elif len(glob.glob(datapath + "/*.npz")) > 0:
            model, masks = create_network_from_npz(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act, num_covariates=num_covariates, 
                                                   mask_order=args.mask_order, one_hot=one_hot)

    # Compile the model with the custom loss function and optimizer
    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer_model,
                  metrics=["accuracy", sensitivity, specificity])

    # Save the model architecture to a file
    with open(resultpath + '/model_architecture.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Set up callbacks for training
    csv_logger = K.callbacks.CSVLogger(resultpath + 'train_log.csv', append=True)
    
    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=1, mode='auto',
                                           restore_best_weights=True)
    save_best_model = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto')
    
    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    
    if os.path.exists(resultpath + '/bestweights_job.h5') and not(args.resume):
        print('Model already Trained')
    elif os.path.exists(resultpath + '/bestweights_job.h5') and args.resume:
        print("load and save weights before resuming")
        shutil.copyfile(resultpath + '/bestweights_job.h5', resultpath + '/weights_before_resuming_' 
                        + datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%p")+'.h5') # save old weights
        log_file = pd.read_csv(resultpath + "/train_log.csv")
        save_best_model = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto', 
                                                      initial_value_threshold=log_file.val_loss.min())
            
        print("Resuming training")
        model.load_weights(resultpath + '/bestweights_job.h5')
        train_generator = TrainDataGenerator(datapath=datapath,
                                             genotype_path=genotype_path,
                                             batch_size=batch_size,
                                             trainsize=int(train_size),
                                             inputsize=inputsize,
                                             epoch_size=args.epoch_size,
                                             one_hot=one_hot)
        history = model.fit_generator(
            generator=train_generator,
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr],
            workers=args.workers,
            use_multiprocessing=multiprocessing,
            validation_data=EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size,
                                          setsize=val_size_train, one_hot=one_hot,
                                          inputsize=inputsize, evalset="validation")

        )
    else:
        print("Start training from scratch")
        train_generator = TrainDataGenerator(datapath=datapath,
                                             genotype_path=genotype_path,
                                             batch_size=batch_size,
                                             trainsize=int(train_size),
                                             inputsize=inputsize,
                                             epoch_size=args.epoch_size,
                                             one_hot=one_hot)
        history = model.fit_generator(
            generator=train_generator,
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr],
            workers=args.workers,
            use_multiprocessing=multiprocessing,
            validation_data=EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size,
                                          setsize=val_size_train, one_hot=one_hot,
                                          inputsize=inputsize, evalset="validation")

        )

    plot_loss_function(resultpath)
    model.load_weights(resultpath + '/bestweights_job.h5')
    print("Finished")
    print("Analysis over the validation set")
    pval = model.predict_generator(
        EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size, setsize=val_size,
                      inputsize=inputsize,one_hot=one_hot,
                      evalset="validation"))
    yval = get_labels(datapath, set_number=2)
    auc_val, confusionmatrix_val = evaluate_performance(yval, pval)
    np.save(resultpath + "/pval.npy", pval)

    print("Analysis over the test set")
    ptest = model.predict_generator(
        EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size, setsize=test_size,
                      inputsize=inputsize, one_hot=one_hot, evalset="test"))
    ytest = get_labels(datapath, set_number=3)
    auc_test, confusionmatrix_test = evaluate_performance(ytest, ptest)
    np.save(resultpath + "/ptest.npy", ptest)

    data = {'Jobid': args.ID,
            'Datapath': str(args.path),
            'genotype_path': str(genotype_path),
            'Batchsize': args.batch_size,
            'Learning rate': args.learning_rate,
            'L1 value': args.L1,
            'L1 act': args.L1_act,
            'patience': args.patience,
            'epoch size': args.epoch_size,
            'epochs': args.epochs,
            'Weight positive class': args.wpc,
            'AUC validation': auc_val,
            'AUC test': auc_test,
            'SlURM_JOB_ID': SlURM_JOB_ID}
    
    pd_summary_row = pd.Series(data)
    pd_summary_row.to_csv(resultpath + "/pd_summary_results.csv")
    
    data['confusionmatrix_val'] = confusionmatrix_val
    data['confusionmatrix_test'] = confusionmatrix_test
    
    with open(resultpath + "results_summary.txt", 'w') as f: 
        for key, value in data.items(): 
            f.write('%s:%s\n' % (key, value))

    if os.path.exists(datapath + "/topology.csv"):
        importance_csv = create_importance_csv(datapath, model, masks)
        importance_csv.to_csv(resultpath + "connection_weights.csv")


def train_regression(args):
    SlURM_JOB_ID = get_SLURM_id()

    # Initialize variables for model and masks
    model = None
    masks = None

    # Setting up paths and parameters from the input arguments
    datapath = args.path
    jobid = args.ID
    lr_opt = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    l1_value = args.L1
    L1_act = args.L1_act
    problem_type = args.problem_type
    patience = args.patience
    one_hot = args.onehot

    # Determine genotype path
    if args.genotype_path == "undefined":
        genotype_path = datapath
    else:
        genotype_path = args.genotype_path

    # Enable mixed precision if specified
    if args.mixed_precision == True:
        use_mixed_precision()

    # Set multiprocessing based on the number of workers specified
    if args.workers > 1:
        multiprocessing = True
    else:
        multiprocessing = False

    # Check data integrity and format
    check_data(datapath=datapath, genotype_path=genotype_path, mode=problem_type)

    # Set up the optimizer for the model
    optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)

    # Calculate dataset sizes for training, validation, and test sets
    train_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 1)
    val_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 2)
    test_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 3)
    num_covariates = pd.read_csv(datapath + "subjects.csv").filter(like='cov_').shape[1]

    # Determine input size for the network
    inputsize = get_inputsize(genotype_path)
    
    # Adjust validation set size based on the epoch size parameter
    val_size_train = val_size

    if args.epoch_size is None:
        args.epoch_size = train_size
    else:
        val_size_train = min(args.epoch_size // 2, val_size)
        print("Using each epoch", args.epoch_size,"randomly selected training examples")
        print("Validation set size used during training is also set to half the epoch_size")


    folder, resultpath = get_paths(args)

    print("jobid =  " + str(jobid))
    print("folder = " + str(folder))
    print("resultpath = " + str(resultpath))
    print("batchsize = " + str(batch_size))
    print("lr = " + str(lr_opt))
    print("L1 = " + str(l1_value))
    print("L1_act = " + str(L1_act))
    print("onehot = " + str(one_hot))
    print("model = " + str(args.network_name))

    
    if args.network_name == "lasso":
        print("lasso network")
        model, masks = lasso(inputsize=inputsize, l1_value=l1_value)
    elif args.network_name == "regression_height":
        print("regression_height network")
        model, masks = regression_height(inputsize=inputsize, l1_value=l1_value)
    elif args.network_name == "gene_network_multiple_filters":
        print("gene_network_multiple_filters network")
        model, masks = gene_network_multiple_filters(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act, regression=True, 
                                                     num_covariates=num_covariates,  filters=args.filters)
    elif args.network_name == "gene_network_snp_gene_filters":
        print("gene_network_snp_gene_filters network")
        model, masks = gene_network_snp_gene_filters(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act, regression=True, 
                                                     num_covariates=num_covariates, filters=args.filters)       
        
        
    else:
        if os.path.exists(datapath + "/topology.csv"):
            model, masks = create_network_from_csv(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act, regression=True, one_hot=one_hot,
                                                   num_covariates=num_covariates)
        elif len(glob.glob(datapath + "/*.npz")) > 0:
            model, masks = create_network_from_npz(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                                   l1_value=l1_value, L1_act =L1_act, regression=True, one_hot=one_hot,
                                                   num_covariates=num_covariates)

    # Compile the model for regression with MSE loss
    model.compile(loss="mse", optimizer=optimizer_model,
                  metrics=["mse"])

    with open(resultpath + '/model_architecture.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Set up callbacks for training
    csv_logger = K.callbacks.CSVLogger(resultpath + 'train_log.csv', append=True)
    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=1, mode='auto',
                                           restore_best_weights=True)
    save_best_model = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto')
    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    

    if os.path.exists(resultpath + '/bestweights_job.h5') and not(args.resume):
        print('Model already trained')
    elif os.path.exists(resultpath + '/bestweights_job.h5') and args.resume:
        print("load and save weights before resuming")
        shutil.copyfile(resultpath + '/bestweights_job.h5', resultpath + '/weights_before_resuming_' 
                        + datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%p")+'.h5') # save old weights
        
        log_file = pd.read_csv(resultpath + "/train_log.csv")
        save_best_model = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto', 
                                                      initial_value_threshold=log_file.val_loss.min())
        print("Resuming training")
        model.load_weights(resultpath + '/bestweights_job.h5')
                
        history = model.fit_generator(
            generator=TrainDataGenerator(datapath=datapath,
                                         genotype_path=genotype_path,
                                         batch_size=batch_size,
                                         trainsize=int(train_size),
                                         inputsize=inputsize,
                                         epoch_size=args.epoch_size,
                                         one_hot=one_hot),
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr],
            workers=args.workers,
            use_multiprocessing=multiprocessing,
            validation_data=EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size,
                                          setsize=val_size_train, inputsize=inputsize, one_hot=one_hot, evalset="validation")
        )
     
    else:
        print("Start training from scratch")
        history = model.fit_generator(
            generator=TrainDataGenerator(datapath=datapath,
                                         genotype_path=genotype_path,
                                         batch_size=batch_size,
                                         trainsize=int(train_size),
                                         inputsize=inputsize,
                                         epoch_size=args.epoch_size,
                                         one_hot=one_hot),
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr],
            workers=args.workers,
            use_multiprocessing=multiprocessing,
            validation_data=EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size,
                                          setsize=val_size_train, inputsize=inputsize, one_hot=one_hot, evalset="validation")
        )

    plot_loss_function(resultpath)
    model.load_weights(resultpath + '/bestweights_job.h5')
    print("Finished")
    print("Analysis over the validation set")
    pval = model.predict_generator(
        EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size, setsize=val_size,
                      evalset="validation", inputsize=inputsize, one_hot=one_hot))
    yval = get_labels(datapath, set_number=2)
    fig, mse_val, explained_variance_val, r2_val = evaluate_performance_regression(yval, pval)
    np.save(resultpath + "/pval.npy", pval)
    fig.savefig(resultpath + "/validation_predictions.png", bbox_inches='tight', pad_inches=0)

    print("Analysis over the test set")
    ptest = model.predict_generator(
        EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size, setsize=test_size,
                      inputsize=inputsize, evalset="test", one_hot=one_hot))
    ytest = get_labels(datapath, set_number=3)
    fig, mse_test, explained_variance_test, r2_test = evaluate_performance_regression(ytest, ptest)
    np.save(resultpath + "/ptest.npy", ptest)
    fig.savefig(resultpath + "/test_predictions.png", bbox_inches='tight', pad_inches=0)

    data = {'Jobid': args.ID,
            'Datapath': str(args.path),
            'genotype_path': str(genotype_path),
            'Batchsize': args.batch_size,
            'Learning rate': args.learning_rate,
            'L1 value': args.L1,
            'L1 act': args.L1_act,
            'patience': args.patience,
            'epoch size': args.epoch_size,
            'epochs': args.epochs,
            'MSE validation': mse_val,
            'MSE test': mse_test,
            'Explained variance val': explained_variance_val,
            'Explained variance test': explained_variance_test,
            'R2_validation': r2_val,
            'R2_test': r2_test,           
            'SlURM_JOB_ID': SlURM_JOB_ID}
    
    pd_summary_row = pd.Series(data)
    pd_summary_row.to_csv(resultpath + "/pd_summary_results.csv")
    
    with open(resultpath + "results_summary.txt", 'w') as f: 
        for key, value in data.items(): 
            f.write('%s:%s\n' % (key, value))

    if os.path.exists(datapath + "/topology.csv"):
        importance_csv = create_importance_csv(datapath, model, masks)
        importance_csv.to_csv(resultpath + "connection_weights.csv")
