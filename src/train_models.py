import tensorflow as tf
import numpy as np
import argparse

from experiments.training_process_factory import TrainingProcessFactory
from experiments.model_factories import CWAEFactory, CWAESamplingFactory
from experiments.args_processors import get_architecture_provider, get_optimizer_factory, get_dataset_loader, prepare_base_dir
from rec_errors import mean_squared_euclidean_norm_reconstruction_error

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str)
parser.add_argument('dataset_name', type=str)
parser.add_argument('epochs_count', type=int)
parser.add_argument('latent_size', type=int)

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=0.001)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--report_after_epoch', type=int, default=5)

parser.add_argument('--save_sampled_images', type=int, default=4)
parser.add_argument('--save_random_train_interpolations', type=int, default=7)
parser.add_argument('--interpolations_step_count', type=int, default=8)
parser.add_argument('--save_random_train_reconstructions', type=int, default=4)

args = parser.parse_args()

dataset_loader = get_dataset_loader(args.dataset_name, args.batch_size)
suffix = f''
base_dir = prepare_base_dir(args)
optimizer_factory = get_optimizer_factory(args.optimizer, args.learning_rate)

rec_error = mean_squared_euclidean_norm_reconstruction_error
model_factories = {
    'cwae': lambda: CWAEFactory(rec_error),
    'cwae_sampling': lambda: CWAESamplingFactory(rec_error)
}

model_name = args.model
factory_method = model_factories[model_name]

tf.reset_default_graph()

dataset = dataset_loader()
architecture_provider = get_architecture_provider(args.dataset_name)

model_factory = factory_method()
experiment_name = f'{model_name}_{args.batch_size}_o{args.optimizer}_lr{args.learning_rate}'
print(experiment_name)

factory = TrainingProcessFactory(dataset, base_dir, model_factory, architecture_provider,
                                 optimizer_factory)
training_process = factory.create(args.latent_size, experiment_name)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as tf_session:
    training_process.start(tf_session)

    tf_session.graph.finalize()

    epochs_count = args.epochs_count
    for current_epoch in range(1, epochs_count+1):
        print(f'Epoch {current_epoch}')
        dataset.initialize_iterator(tf_session)

        training_process.next_epoch(current_epoch)

        if (current_epoch == 1 or current_epoch % args.report_after_epoch == 0 or current_epoch == epochs_count):
            training_process.evaluate(current_epoch)

            model = training_process.get_model()

            z_dim = args.latent_size
            if args.save_sampled_images > 0:
                latents = np.random.multivariate_normal(np.zeros(z_dim), np.identity(z_dim),
                                                        args.save_sampled_images * args.save_sampled_images)

                images = list()
                for i, latent in enumerate(latents):
                    image = model.decode(tf_session, [latent])[0]
                    images.append(image)
                training_process.save_image(f'sampled.png', images, 
                                            size=(args.save_sampled_images, args.save_sampled_images))

            random_train_rec_count = args.save_random_train_reconstructions
            if args.save_random_train_reconstructions > 0:
                print('Ploting random train reconstructions')
                train_indexes = list(np.random.randint(0, dataset.train_images_count,
                                     random_train_rec_count*random_train_rec_count))
                reconstruction_images = list()
                for i, train_index in enumerate(train_indexes):
                    input_image = dataset.train_images[train_index]
                    _, decoded_images = model.encode_decode(tf_session, [input_image])
                    reconstruction_images.extend([input_image, decoded_images[0]])
                training_process.save_image(f'reconstructions.png', reconstruction_images,
                                            size=(random_train_rec_count, random_train_rec_count))

            interpolations_step_count = args.interpolations_step_count
            interpolations_count = args.save_random_train_interpolations
            if interpolations_count > 0:
                print('Plotting random train interpolations')
                interpolation_indexes = list(np.reshape(np.random.randint(0, dataset.train_images_count,
                                                                          interpolations_count * 2), [-1, 2]))
                images = list()
                for i, train_indexes in enumerate(interpolation_indexes):
                    first_index = train_indexes[0]
                    second_index = train_indexes[1]

                    first_image = dataset.train_images[first_index]
                    second_image = dataset.train_images[second_index]

                    first_latent, first_image_decoded = model.encode_decode(tf_session, [first_image])
                    first_latent = first_latent[0]
                    first_image_decoded = first_image_decoded[0]
                    second_latent, second_latent_decoded = model.encode_decode(tf_session, [second_image])
                    second_latent = second_latent[0]
                    second_latent_decoded = second_latent_decoded[0]
                    images.extend([first_image, first_image_decoded])

                    latent_step = (second_latent - first_latent) / (interpolations_step_count + 1)
                    for j in range(interpolations_step_count):
                        next_latent = first_latent + (j + 1) * latent_step
                        next_image = model.decode(tf_session, [next_latent])[0]
                        images.append(next_image)

                    images.extend([second_latent_decoded, second_image])

                training_process.save_image(f'interpolations.png', images, size=(interpolations_count,
                                                                                 interpolations_step_count + 4))

    training_process.end()
