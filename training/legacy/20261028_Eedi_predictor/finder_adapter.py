from original_core import *
def fit_finders(data, rows, valid_rows, description, config, seed):
    tf.keras.utils.set_random_seed(seed)
    n_people, n_items = description["people"], description["items"]
    size = config["latent_size"]
    initial_users = np.random.default_rng(seed).random((n_people, size)).astype(np.float32)
    initial_items = np.random.default_rng(seed + 1000).random((n_items, size)).astype(np.float32)
    zeros = np.zeros((n_items, 1), np.float32)
    user_finder = build_finder_model(initial_users.shape, zeros.shape, initial_items.shape, True, False)
    item_finder = build_finder_model(initial_users.shape, zeros.shape, initial_items.shape, False, True)
    for finder in [user_finder, item_finder]:
        finder.user_latents_layer.set_value(initial_users)
        finder.item_features_layer.set_value(zeros)
        finder.item_latents_layer.set_value(initial_items)
        finder.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss=BatchBinaryCrossentropy(),
                       metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")])
    training = ResponseBatches(data, rows, config["batch_size"], True, seed)
    validation = ResponseBatches(data, valid_rows, config["batch_size"], False, seed)
    schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=2, min_lr=1e-5)
    histories = []
    for cycle in range(config["finder_cycles"]):
        user_finder.item_latents_layer.set_value(item_finder.item_latents_layer.get_value())
        for label, finder in [("ULD", user_finder), ("ILD", item_finder)]:
            if label == "ILD":
                item_finder.user_latents_layer.set_value(user_finder.user_latents_layer.get_value())
            history = finder.fit(training, validation_data=validation, epochs=config["finder_subepochs"], verbose=0,
                                 callbacks=[schedule, TrainingProgress(f"cycle {cycle+1} {label}")])
            histories.append({"phase":label, "cycle":cycle+1, "values":history.history})
    final_users = user_finder.user_latents_layer.get_value()
    final_items = item_finder.item_latents_layer.get_value()
    user_finder.item_latents_layer.set_value(final_items)
    del item_finder
    gc.collect()
    return {'final_users':final_users, 'final_items':final_items, 'histories':histories}
