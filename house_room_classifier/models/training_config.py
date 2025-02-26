# class TrainingConfig:
#     def __init__(
#             self,
#             optimizer='adam',
#             learning_rate=0.0001,
#             loss='sparse_categorical_crossentropy',
#             epochs=10,
#             early_stopping_patience=50,
#             learning_rate_decay=0.9,
#             learning_rate_decay_steps=100,
#             use_data_augmentation=True
#     ):
#         self.optimizer=optimizer
#         self.learning_rate=learning_rate
#         self.loss=loss
#         self.epochs=epochs
#         self.early_stopping_patience=early_stopping_patience
#         self.learning_rate_decay=learning_rate_decay
#         self.learning_rate_decay_steps=learning_rate_decay_steps
#         self.use_data_augmentation=use_data_augmentation
        
class TrainingConfig:
    def __init__(
            self,
            optimizer='sgd',  # Changed from 'adam' to 'sgd'
            learning_rate=0.01,  # Increased learning rate for SGD
            momentum=0.9,  # Added momentum for stable updates
            loss='sparse_categorical_crossentropy',
            epochs=20,  # Increased epochs to allow convergence
            early_stopping_patience=10,  # Adjusted to prevent early stopping
            learning_rate_decay=0.95,  # Slightly lower decay for smooth learning
            learning_rate_decay_steps=100,
            use_data_augmentation=True
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum  # Store momentum parameter
        self.loss = loss
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.use_data_augmentation = use_data_augmentation