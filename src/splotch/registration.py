"""registration.py."""

import logging
from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, grad, jit, random
from jax.example_libraries.optimizers import OptimizerState, adagrad

from splotch.utils import SplotchInputData

KeyArray = Array


def register(
    key: KeyArray,
    splotch_input_data: SplotchInputData,
    num_steps: int = 10_000,
    aars_of_interest: list[str] | None = None,
) -> None:
    """Register tissue sections using AARs.

    Args:
        key: PRNGKey.
        splotch_input_data: TBA.
        num_steps: TBA.
        aars_of_interest: TBA.
    """
    if aars_of_interest is None:
        aars_of_interest = list(splotch_input_data.annotation_data.columns)

    aars = [
        list(splotch_input_data.annotation_data.columns).index(aar)
        for aar in aars_of_interest
    ]

    unique_tissue_sections = splotch_input_data.metadata.index.get_level_values(
        "tissue_section"
    ).unique()
    coordinates_df = splotch_input_data.metadata[["x", "y"]]
    annotations_df = pd.Series(
        np.argmax(splotch_input_data.annotation_data, 1),
        index=splotch_input_data.annotation_data.index,
    )

    coordinates = [
        coordinates_df.loc[pd.IndexSlice[:, tissue_section, :], :].values
        for tissue_section in unique_tissue_sections
    ]
    annotations = np.hstack(
        [
            annotations_df.loc[pd.IndexSlice[:, tissue_section, :]].values
            for tissue_section in unique_tissue_sections
        ]
    )

    # move to origin
    coordinates = [x - np.mean(x, axis=0, keepdims=True) for x in coordinates]

    logging.info("Start the tissue section registration")
    coordinates_registered = register_tissue_sections(
        key, coordinates, annotations, num_steps, aars
    )

    logging.info("Register the consensus spot cloud")
    coordinates_registered = register_consensus(
        coordinates_registered,
        annotations,
        aars,
    )

    splotch_input_data.metadata["x_registration"] = coordinates_registered[:, 0]
    splotch_input_data.metadata["y_registration"] = coordinates_registered[:, 1]


def register_tissue_sections(
    key: KeyArray,
    x: list[np.ndarray],
    y: np.ndarray,
    num_steps: int,
    aars_of_interest: list[int],
) -> np.ndarray:
    """Registed individual tissue sections.

    Args:
        key: PRNGKey.
        x: TBA.
        y: TBA.
        num_steps: TBA.
        aars_of_interest: TBA.

    Returns:
        Registered coordinates.
    """
    aar_indices = [np.where(y == aar)[0] for aar in aars_of_interest]
    uti_indices = [np.triu_indices(np.sum(y == aar), k=1) for aar in aars_of_interest]

    def transform(param: Array, x: list[np.ndarray]) -> Array:
        theta_params = param[0, :]
        delta_params = params[1:, :].T
        return jnp.hstack(
            [
                jnp.dot(
                    jnp.array(
                        [
                            [jnp.cos(theta), -jnp.sin(theta)],
                            [jnp.sin(theta), jnp.cos(theta)],
                        ]
                    ),
                    point.T,
                )
                + delta[:, None]
                for theta, delta, point in zip(theta_params, delta_params, x)
            ]
        ).T

    def f(param: Array, x: list[np.ndarray]) -> Array:
        def loss(x: Array) -> Array:
            def helper(x: Array, uti: tuple[np.ndarray, np.ndarray]) -> Array:
                dr = x[uti[0], :] - x[uti[1], :]
                return jnp.sum(jnp.sqrt(jnp.sum(dr * dr, axis=1)))

            return jnp.sum(
                jnp.asarray(
                    [
                        helper(x[aar_indices[aar], :], uti_indices[aar])
                        for aar in range(len(aars_of_interest))
                    ]
                )
            )

        return loss(transform(param, x))

    loss = partial(f, x=x)

    opt_init, opt_update, get_params = adagrad(step_size=0.1, momentum=0.9)

    @jit
    def step(i: int, opt_state: OptimizerState) -> OptimizerState:
        params = get_params(opt_state)
        g = grad(loss)(params)
        return opt_update(i, g, opt_state)

    params = jnp.vstack(
        (
            random.uniform(key, (1, len(x)), minval=-jnp.pi, maxval=jnp.pi),
            np.zeros((2, len(x))),
        )
    )
    prev_value = loss(params)
    logging.info("Iteration 0: loss = %f", prev_value)
    opt_state = opt_init(params)
    for i in range(num_steps):
        opt_state = step(i, opt_state)
        if i > 0 and i % 100 == 0:
            params = get_params(opt_state)
            curr_value = loss(params)
            logging.info("Iteration %d: loss = %f", i + 1, curr_value)

            if jnp.isclose(prev_value, curr_value):
                logging.info("Converged after %d iterations", i + 1)
                params = get_params(opt_state)
                return np.asarray(transform(params, x))

            prev_value = curr_value

    logging.warning("Not converged after %d iterations", i + 1)
    params = get_params(opt_state)
    return np.asarray(transform(params, x))


def register_consensus(
    x: np.ndarray, y: np.ndarray, aars_of_interest: list[int]
) -> np.ndarray:
    """Register consensus point cloud.

    Args:
        x: Coordinates.
        y: Annotations.
        aars_of_interest: AARs of interest.

    Returns:
        Registered coordinates.
    """

    def rotate(theta: float, x: np.ndarray) -> np.ndarray:
        """Rotate points.

        Args:
            theta: Angle.
            x: Points.

        Returns:
            Rotated points.
        """
        return np.dot(
            np.asarray(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            ),
            x.T,
        ).T

    values = np.zeros((len(aars_of_interest), 2))
    for aar in aars_of_interest:
        cov_matrix = np.cov(x[y == aar, :], rowvar=False)
        u, v = np.linalg.eigh(cov_matrix)
        values[aar, :] = [u[-1], np.arctan2(v[1, -1], v[0, -1])]

    theta = values[np.argmax(values[:, 0]), 1]

    x_registered = rotate(-theta, x)
    return x_registered - np.mean(x_registered, 0, keepdims=True)
