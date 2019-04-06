import pandas as pd
import random
import numpy as np

class Ranking(object):
    def __init__(self, test_size=0.1, data_size=None):
        self.num_prior = 10
        self.positive_negative_rating_path = 'result.csv'

    def prepare_rating_data(self, file_path):

        self.data = pd.read_csv(file_path, dtype={'item_id': object, 'user_id': object}).sort_values('timestamp')
        df = self.data
        df.dropna(subset=['item_id', 'user_id'], inplace=True)

        results = []
        for user_id, group in df.sort_values(
            ['user_id', 'timestamp'], ascending=[True, False]
        ).groupby('user_id'):
          group = group.reset_index()
          positive = None

          for index, row in group.iterrows():
              # print(index)
              if row.rating_type not in [20, 90]:
                positive = row

                num_prior = self.num_prior
                low = max(0, index - num_prior)
                priors = group.drop_duplicates(subset=['user_id', 'item_id'])[low:index]

                result_positive_dict = {
                  'user_id': user_id,
                  'item_id': positive.item_id,
                  'prior_ids': ','.join(priors.item_id),
                  'target': 1
                }
                results.append(result_positive_dict)
                # 20, 90 = negative
                positives = group[(group.index>=index) & (~group.rating_type.isin([20, 90]))][:10]

                num_negative = 4

                for i in range(num_negative):
                  index_sample = random.sample(range(index+1), 1)[0]
                  sample = group.iloc[index_sample]

                  low = max(0, index_sample - num_prior)

                  try_count = 5
                  for _ in range(try_count):
                      if sample.rating_type not in [20, 90] or sample.item_id in positives.item_id:
                        index_sample = random.sample(range(index+1), 1)[0]
                        sample = group.iloc[index_sample]

                  low = max(0, index_sample - num_prior)
                  priors = group.drop_duplicates(subset=['user_id', 'item_id'])[low:index_sample]

                  negative = sample
                  result_negative_dict = {
                    'user_id': user_id,
                    'item_id': negative.item_id,
                    'prior_ids': ','.join(priors.item_id),
                    'target': 0
                  }

                  results.append(result_negative_dict)

          if positive is None:

            group = group.drop_duplicates(
              subset=['user_id', 'item_id'])
            n = min(len(group), self.num_prior + 1)

            group = group.sample(n)

            result_negative_dict = {
                'user_id': user_id,
                'item_id': group.tail(1)['item_id'].iloc[0],
                'prior_ids': ','.join(group.item_id[:-1]),
                'target': 0
              }

            results.append(result_negative_dict)

        df_result = pd.DataFrame(results, columns=['item_id', 'prior_ids', 'target', 'user_id'])
        df_result = self.apply_prior_ids_pad(df_result)
        df_result.to_csv(self.positive_negative_rating_path, sep=',', encoding='utf-8', index=False)

        self.df = df_result
        return df_result

    def apply_prior_ids_pad(self, df):

      def pad(x):

        x = x.strip()
        result = [a for a in x.split(',') if a]
        result = result + ['0'] * (self.num_prior - len(result))

        return result
      df['prior_ids'] = df['prior_ids'].apply(pad)

      return df


ranking = Ranking()
ranking.prepare_rating_data('ratings_sample.csv')