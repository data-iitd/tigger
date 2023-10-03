import os
import re
import pickle
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

try:
    import matplotlib.pyplot as plt
except:
    pass

class FlowNet():
    def __init__(self, config_path, config_dict):
        self.config_path = config_path + config_dict['flownet_config_path']
        for key, val in config_dict.items():
            setattr(self, key, val)
 
        self.model = None
        self.trainable_distribution = None

        
        os.makedirs(self.config_path, exist_ok=True)
        
    def make_masked_autoregressive_flow(self):
        made = tfb.AutoregressiveNetwork(hidden_units=self.hidden_units, 
                                        params=2, event_shape=[self.event_dim], activation=self.activation)
        return tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)

    def init_model(self):
        # start is independent normal distribution with event dim equal to embed+node dim
        normal_distributions = tfd.Normal(loc=[0]*self.event_dim, scale=[1]*self.event_dim)
        base_distribution = tfd.Independent(normal_distributions, reinterpreted_batch_ndims=1)
        
        # create trainable distribution
        bijectors = []
        for i in range(self.number_of_bijectors):
            bijectors.append(self.make_masked_autoregressive_flow())
            permutation = list(range(self.event_dim))
            if i%2 != 0:
                permutation.reverse()
            bijectors.append(tfb.Permute(permutation=permutation))
            
        flow_bijector = tfb.Chain(bijectors[:-1])
        trainable_distribution = tfd.TransformedDistribution(
            base_distribution,
            flow_bijector)
        self.trainable_distribution = trainable_distribution
        
        x_ = Input(shape=(self.event_dim,), dtype=tf.float32)
        log_prob_ = trainable_distribution.log_prob(x_)
        self.model = Model(x_, log_prob_)
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
                    loss=lambda _, log_prob: -log_prob)
    
  
    def train_model(self, x_data):
        ns = x_data.shape[0]
        batch_size = min(ns, self.batch_size)

        # Display the loss every n_disp epoch
        epoch_callback = LambdaCallback(
            on_epoch_end=lambda cur_epoch, logs: 
                            print('\n Epoch {}/{}'.format(cur_epoch+1, self.epoch, logs),
                                '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                        if cur_epoch % self.n_disp == 0 else False 
        )


        history = self.model.fit(x=x_data,
                            y=np.zeros((ns, 0), dtype=np.float32),
                            batch_size=batch_size,
                            epochs=self.epoch,
                            validation_split=0.2,
                            shuffle=True,
                            verbose=False,
                            callbacks=[epoch_callback])
        return history
        
    def sample_model(self, size, name=None):
        samples = self.trainable_distribution.sample(size)    
        sample_df = pd.DataFrame(samples)
        if not name:
            name = self.config_path + "synth_nodes.parquet"
        sample_df.to_parquet(name)
        
    def train(self, embed, node_attr):
        x_data = self.prep_data(embed, node_attr)
        
        if not self.model:
            self.init_model()
            
        history = self.train_model(x_data)
        name = self.save_model()
        
        if self.verbose >= 2:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.legend()
            # plt.yscale("log")
            plt.title("loss of flownet")
            plt.show()        
            
        return (name, history)
        
    def prep_data(self, embed, node_attr):
        
        x_data = embed.join(node_attr, how='inner')
        self.event_dim = x_data.shape[1]
        
        assert embed.shape[0] == x_data.shape[0]
        if node_attr.shape[0] != x_data.shape[0]:
            warnings.warn("not all nodes are included! are there nodes without edges?")
        
        return x_data
    
    def save_model(self):
        
        files = os.listdir(self.config_path)
        version = 0
    
        for f in files:
            f_list = re.split("[_|.]", f)
            if f_list[0]=='flowmodel' and int(f_list[1])>version:
                version = int(f_list[1])
        
        if not self.overwrite:
            version = version + 1
        
        version = max(version, 1)
                    
    
        name = "flowmodel_" + str(version) + ".pickle"
        weights = self.model.get_weights()
        pickle.dump(weights, open(self.config_path + name, "wb"))
        # self.model.save(self.config_path + name)
        return name
        
    def load_model(self, model_name):
        if not self.model:
            self.init_model()
            
        weights = pickle.load(open(self.config_path + model_name, "rb"))
        self.model.set_weights(weights)
        
    def lin_grid_search(self, grid_dict, embed, node_attr):
        dim = list(grid_dict.keys())[0]
        vals = grid_dict[dim]
        res = {}
        
        for val in vals:
            self.model = None
            setattr(self, dim, val)
            name, hist = self.train(embed, node_attr)
            run = {
                'name': name,
                'hist': hist.history,
                'dim': dim,
                'val': val,
                "loss": np.mean(hist.history['loss'][-4:]),
                "val_loss": np.mean(hist.history['val_loss'][-4:]),
            }
            res[val]=run
            
        if self.verbose>=2:
            self.plot_grid(res)
        return res
    
    def plot_grid(self, res):
        losses = []
        val_losses = []
        for k, v in res.items():
            losses.append(v['loss'])
            val_losses.append(v['val_loss'])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        keys = [str(k) for k in res.keys()]
        ax1.bar(keys, losses, label='loss')
        ax1.bar(keys, val_losses, label='val_loss')
        for k, v in res.items():
            ax2.plot(v['hist']['val_loss'], label=str(k))
        ax2.legend()
        print(f"loss: {losses}")
        print(f"val loss: {val_losses}")
            
        
        