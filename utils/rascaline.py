import copy
import numpy as np
import torch

from rascaline import SphericalExpansion

from aml_storage import Labels, Block, Descriptor


class RascalineSphericalExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames) -> Descriptor:
        max_radial = self._hypers["max_radial"]
        max_angular = self._hypers["max_angular"]

        calculator = SphericalExpansion(**self._hypers)
        descriptor = calculator.compute(frames)

        old_samples = descriptor.samples
        old_gradient_samples = descriptor.gradients_samples
        values = descriptor.values.reshape(descriptor.values.shape[0], -1, max_radial)

        species = np.unique(old_samples[["species_center", "species_neighbor"]])
        sparse = Labels(
            names=["spherical_harmonics_l", "center_species", "neighbor_species"],
            values=np.array(
                [
                    [l, species_center, species_neighbor]
                    for l in range(max_angular + 1)
                    for species_center, species_neighbor in species
                ],
                dtype=np.int32,
            ),
        )

        features = Labels(
            names=["n"],
            values=np.array([[n] for n in range(max_radial)], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(max_angular + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        if descriptor.gradients is not None:
            has_gradients = True
            gradients = descriptor.gradients.reshape(
                descriptor.gradients.shape[0], -1, max_radial
            )
        else:
            has_gradients = False

        blocks = []
        for sparse_i, (l, center_species, neighbor_species) in enumerate(sparse):
            centers = np.unique(
                old_samples[old_samples["species_center"] == center_species][
                    ["structure", "center"]
                ]
            )
            center_map = {tuple(center): i for i, center in enumerate(centers)}

            block_data = np.zeros((len(centers), 2 * l + 1, max_radial))

            mask = np.logical_and(
                old_samples["species_center"] == center_species,
                old_samples["species_neighbor"] == neighbor_species,
            )
            for sample_i in np.where(mask)[0]:
                new_sample_i = center_map[
                    tuple(old_samples[sample_i][["structure", "center"]])
                ]
                block_data[new_sample_i, :, :] = values[sample_i, lm_slices[l], :]

            samples = Labels(
                names=["structure", "center"],
                values=np.array(
                    [[structure, center] for structure, center in center_map.keys()],
                    dtype=np.int32,
                ),
            )
            components = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            block_gradients = None
            gradient_samples = None
            if has_gradients:
                gradient_samples = []
                block_gradients = []
                for sample_i in np.where(mask)[0]:
                    gradient_mask = old_gradient_samples["sample"] == sample_i

                    new_sample_i = center_map[
                        tuple(old_samples[sample_i][["structure", "center"]])
                    ]

                    for grad_index in np.where(gradient_mask)[0]:
                        block_gradients.append(
                            gradients[grad_index : grad_index + 1, lm_slices[l], :]
                        )

                        structure = old_samples[sample_i]["structure"]
                        atom, spatial = old_gradient_samples[grad_index][
                            ["atom", "spatial"]
                        ]
                        gradient_samples.append(
                            (new_sample_i, structure, atom, spatial)
                        )

                if len(gradient_samples) != 0:
                    block_gradients = np.concatenate(block_gradients)
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom", "spatial"],
                        values=np.vstack(gradient_samples).astype(np.int32),
                    )
                else:
                    block_gradients = np.zeros(
                        (0, components.shape[0], features.shape[0])
                    )
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom", "spatial"],
                        values=np.zeros((0, 4), dtype=np.int32),
                    )

            block = Block(
                values=block_data,
                samples=samples,
                components=components,
                features=features,
            )

            if block_gradients is not None:
                block.add_gradient("positions", gradient_samples, block_gradients)

            blocks.append(block)

        return Descriptor(sparse, blocks)
