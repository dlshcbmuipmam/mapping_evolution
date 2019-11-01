import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sns
from im_tutorials.data.cordis import cordis_table

# merge organisations, project_organisations, projects
cordis_orgs_df = cordis_table('organisations')
cordis_project_orgs_df = cordis_table('project_organisations')
cordis_projects_df = cordis_table('projects')
cordis_full_df = cordis_project_orgs_df.merge(
    cordis_projects_df, left_on='project_rcn', right_on='rcn', how='left'
)
cordis_full_df = cordis_full_df.merge(
    cordis_orgs_df, left_on='organization_id', right_on='id', how='left'
)
cordis_full_df = cordis_full_df [(cordis_full_df['activity_type'] == 'Private for-profit entities (excluding Higher or Secondary Education Establishments)')
  & (cordis_full_df['framework'] == 'H2020')]

cordis_full_df.head()
cordis_full_df.columns

# Add num_success, group_multiple_success
cordis_full_df_sorted = cordis_full_df.sort_values(by=['organization_id', 'start_date_code'])
cordis_full_df_sorted = cordis_full_df_sorted.reset_index(drop=True)
cordis_full_df_sorted['num_success'] = cordis_full_df_sorted.groupby(['organization_id']).cumcount()
cordis_full_df_sorted['multiple_success'] = cordis_full_df_sorted.groupby(['organization_id'])['num_success'].transform('max')
cordis_full_df_sorted['group_multiple_success'] = (cordis_full_df_sorted['multiple_success'] > 0) * 1
cordis_full_df_sorted.columns

# Add funded_under_title
def fundedUnder2Title(x):
    return x[0]['title']

cordis_full_df_sorted['funded_under_title'] = cordis_full_df_sorted['funded_under'].apply(fundedUnder2Title)


# Add funding_scheme_mean_ec_contribution
cordis_full_df_sorted['funding_scheme_mean_ec_contribution'] = cordis_full_df_sorted.groupby(['funding_scheme'])['ec_contribution'].transform(np.mean)

# Add funding_scheme_total_ec_contribution
cordis_full_df_sorted['funding_scheme_total_ec_contribution'] = cordis_full_df_sorted.groupby(['funding_scheme'])['ec_contribution'].transform(np.sum)

# Add importance
cordis_full_df_sorted['importance'] = cordis_full_df_sorted['contribution'] / cordis_full_df_sorted['ec_contribution']

# Add num_of_partners
cordis_full_df_sorted['num_of_partners'] = cordis_full_df_sorted.groupby(['project_rcn'])['organization_id'].transform('count')

# Add funding_country_mean_ec_contribution
cordis_full_df_sorted['country_mean_ec_contribution'] = cordis_full_df_sorted.groupby(['country_code'])['ec_contribution'].transform(np.mean)

# Add funding_country_total_ec_contribution
cordis_full_df_sorted['country_total_ec_contribution'] = cordis_full_df_sorted.groupby(['country_code'])['ec_contribution'].transform(np.sum)

# Add funded_under_mean_ec_contribution
cordis_full_df_sorted['funded_under_mean_ec_contribution'] = cordis_full_df_sorted.groupby(['funded_under_title'])['ec_contribution'].transform(np.mean)

# Add funded_under_total_ec_contribution
cordis_full_df_sorted['funded_under_total_ec_contribution'] = cordis_full_df_sorted.groupby(['funded_under_title'])['ec_contribution'].transform(np.sum)

# Add partner_max_multiple_success
cordis_full_df_sorted['max_success_partner'] = cordis_full_df_sorted.groupby(['project_rcn'])['multiple_success'].transform('max')

# Add "CLOSED","ONGOING","SIGNED","TERMINATED"
cordis_full_df_sorted = pd.concat([cordis_full_df_sorted, pd.get_dummies(cordis_full_df_sorted['status'])], axis=1)

# Add participant, coordinator, partner
cordis_full_df_sorted = pd.concat([cordis_full_df_sorted, pd.get_dummies(cordis_full_df_sorted['type'])], axis=1)

cordis_full_df_sorted.columns

from im_tutorials.features.text_preprocessing import clean_and_tokenize
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter
import re
# ' '.join(clean_and_tokenize('hi there i am george and i like docs'))

cordis_full_df_sorted['tokenize_title'] = cordis_full_df_sorted['title'].apply(clean_and_tokenize)
cordis_full_df_sorted['tokenize_title'] = cordis_full_df_sorted['tokenize_title'].apply(' '.join)
cordis_full_df_sorted['tokenize_title'].head()

cordis_full_df_sorted['tokenize_objective'] = cordis_full_df_sorted['objective'].str.replace("<br/>", "").apply(clean_and_tokenize)
cordis_full_df_sorted['tokenize_objective'] = cordis_full_df_sorted['tokenize_objective'].apply(' '.join)
cordis_full_df_sorted['tokenize_objective'].head()

cordis_full_df_sorted['tokenize_text'] = cordis_full_df_sorted['tokenize_title'] + cordis_full_df_sorted['tokenize_objective']
cordis_full_df_sorted['tokenize_text'].head()

token_counts = Counter(chain(*[x.split(' ') for x in cordis_full_df_sorted['tokenize_text']]))
token_counts.most_common(40)

tfidf_vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, stop_words='english')
tfidf_vecs = tfidf_vectorizer.fit_transform(cordis_full_df_sorted['tokenize_text'])

# n_component should be somewhere in the middle
svd = TruncatedSVD(n_components=50, algorithm='arpack', n_iter=7, random_state=42)
svd_vecs = svd.fit_transform(tfidf_vecs)
svd_vecs.shape


clf = KMeans(n_clusters=20, random_state=0)
clf.fit(svd_vecs)
species = clf.predict(svd_vecs)
species.shape

cordis_full_df_sorted['text_species'] = species

cordis_full_df_sorted['text_species'].value_counts()

index = np.random.randint(low = 0, high = species.shape[0] - 1, size = 2000)
svd_vecs[index].shape

tsne = TSNE(n_components=2)
tsne_vecs = tsne.fit_transform(svd_vecs[index])
tsne_vecs.shape

cordis_full_df_sorted = pd.concat([cordis_full_df_sorted, pd.get_dummies(cordis_full_df_sorted['text_species'], prefix="text")], axis=1)

input_var_cont = ['contribution',
                  'ec_contribution',
                  'total_cost',
                  'funding_scheme_mean_ec_contribution',
                  'funding_scheme_total_ec_contribution',
                  'importance',
                  'num_of_partners',
                  'country_mean_ec_contribution',
                  'country_total_ec_contribution',
                  'funded_under_mean_ec_contribution',
                  'funded_under_total_ec_contribution',
                  'max_success_partner'
                 ]

input_var_species = ["text_" + str(i) for i in range(20)]

input_var_disc = ["CLOSED",
                 "ONGOING",
                 "SIGNED",
                 "TERMINATED",
                 "participant",
                 "coordinator",
                 "partner"
                 ]

output_var = ['group_multiple_success']

all_var = input_var_cont + input_var_species + input_var_disc + output_var

cordis_full_df_filtered_num_success = cordis_full_df_sorted[cordis_full_df_sorted['num_success'] == 0]
cordis_full_df_group_multiple_success = cordis_full_df_filtered_num_success[all_var]

normalized_df_group_multiple_success = (cordis_full_df_group_multiple_success - cordis_full_df_group_multiple_success.min())/(cordis_full_df_group_multiple_success.max() - cordis_full_df_group_multiple_success.min())
normalized_df_group_multiple_success.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as ss
from sklearn.metrics import classification_report

X, y = normalized_df_group_multiple_success[input_var_cont + input_var_species + input_var_disc], normalized_df_group_multiple_success['group_multiple_success']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,  # Iris is a pretty easy task so we make it a little harder
    shuffle=True,
    random_state=42,
)


