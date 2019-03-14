import argparse
import pandas as pd
import itertools
import glob
import json

import tensorflow as tf

from ..tf_records_management import save_rows_to_tf_record_file, make_sequential_feature
from ..utils import serialize, deserialize


def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_clicks_json_path_regex', default='',
        help='Input path of the clicks JSON files.')

    parser.add_argument(
        '--chunk_size', type=int, default=100000
    )

    parser.add_argument(
        '--id_map_path', default='',
        help='Input path of article ids registry'
    )

    parser.add_argument(
        '--output_sessions_tfrecords_path', default='',
        help='Output path for TFRecords generated with user sessions')

    return parser


def load_sessions_by_hour(clicks_file_path):
    def to_list(series):
        return list(series)

    clicks_hour_df = pd.read_csv(clicks_file_path)
    # Ensuring that sessions are chronologically ordered
    clicks_hour_df.sort_values(['session_start', 'click_timestamp'], inplace=True)
    sessions_by_hour_df = clicks_hour_df.groupby('session_id').agg({'user_id': min,
                                                                    'session_start': min,
                                                                    'session_size': min,
                                                                    'click_article_id': to_list,
                                                                    'click_timestamp': to_list,
                                                                    'click_environment': to_list,
                                                                    'click_deviceGroup': to_list
                                                                    }
                                                                   ).reset_index()
    return sessions_by_hour_df


def load_id_map(id_map_path):
    with open(id_map_path) as lines:
        return {int(id): idx for idx, id in enumerate(lines)}


def load_sessions_json(files, id_map):
    for file_path in files:
        with open(file_path) as the_input:
            for line in the_input:
                row = json.loads(line)
                page_ids = [id_map[page_id] for page_id in row['pageIds'] if page_id in id_map]
                if len(page_ids) < 5:
                    continue

                yield dict(
                    page_ids=page_ids
                )


def make_sequence_example(row):
    # idx, fields = row

    context_features = {
        # 'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['user_id']])),
        # 'session_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['session_id']])),
        # 'session_start': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['session_start']])),
        'session_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(row["page_ids"])]))
    }
    #
    context = tf.train.Features(feature=context_features)

    sequence_features = {
        # 'event_timestamp': make_sequential_feature(fields["click_timestamp"]),
        # Categorical features
        'item_clicked': make_sequential_feature(row["page_ids"]),
        # 'environment': make_sequential_feature(fields["click_environment"]),
        # 'deviceGroup': make_sequential_feature(fields["click_deviceGroup"]),
    }

    sequence_feature_lists = tf.train.FeatureLists(feature_list=sequence_features)

    return tf.train.SequenceExample(feature_lists=sequence_feature_lists,
                                    context=context
                                    )


def main():
    parser = create_args_parser()
    args = parser.parse_args()

    print('Loading sessions by hour')
    clicks_hour_files = sorted(glob.glob(args.input_clicks_json_path_regex))

    id_map = load_id_map(args.id_map_path)

    print('Exporting sessions by hour to TFRecords: {}'.format(args.output_sessions_tfrecords_path))
    # Exporting a TFRecord for each CSV clicks file (one by hour)

    sessions = load_sessions_json(clicks_hour_files, id_map)
    index = 0
    while True:
        rows = list(itertools.islice(sessions, args.chunk_size))
        if not rows:
            break

        save_rows_to_tf_record_file(
            rows,
            make_sequence_example,
            export_filename=args.output_sessions_tfrecords_path.replace('*', '{0:03d}').format(
                index))

        index += 1

        if index % 10 == 0:
            print('Exported {} TFRecord files'.format(index))


if __name__ == '__main__':
    main()
