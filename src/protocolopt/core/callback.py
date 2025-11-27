class Callback:
    def on_train_start(self, simulation_object): pass
    def on_epoch_start(self, simulation_object, epoch): pass
    def on_epoch_end(self, simulation_object, sim_dict, loss_values, epoch): pass
    def on_train_end(self, simulation_object, sim_dict, coeff_grid, epoch): pass


