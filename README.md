# CLIP AUDIT
The purpose of this repo is to collect activations when auditing CLIP models.

We focus on TinyCLIP although the code should be model agnostic.

**Order of Operations** 
1. `get_neuron_indices.py`: Get random neuron indices from that model. DONE.
2. `test_model_accuracy.py`: Test the accuracy of the model on your dataset. DONE.
3. `cache_activations.py`: Cache activations and image indices for those neurons
4. `plot_all_neurons_histogram.py`: Define activation intervals for each layer, both raw values and SD values. Also get intervals.
    4a. `get_intervals.py`: Get the percentiles per activation.
5. `sample_images_from_interval.py`: Get corresponding heatmaps for those n image indices per interval and save!