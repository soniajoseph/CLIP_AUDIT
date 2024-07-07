# CLIP AUDIT
The purpose of this repo is to collect activations when auditing CLIP models.

We focus on TinyCLIP although the code should be model agnostic.

**Order of Operations** 
1. `get_neuron_indices.py`: Get random neuron indices from that model
2. `test_model_accuracy.py`: Test the accuracy of the model on your dataset
3. `cache_activations.py`: Cache activations and image indices for those neurons
4. `get_activation_intervals.py`: Define activation intervals and randomly sample n image indices from those intervals
5. `save_heatmaps.py`: Get corresponding heatmaps for those n image indices per interval and save!