import csv
import json

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request
from flask import render_template
from scipy.cluster import hierarchy as sch
from scipy.cluster.hierarchy import fcluster, leaders, centroid
from scipy.cluster.hierarchy import linkage, dendrogram, ward, cut_tree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer





def print_cut_tree_features(cut_tr, names):

    for element in cut_tr:
        print(element[0], names[element[0]])

    pass


def get_centroid_feature_for_cluster(descriptors):
    # print("descriptors from getting centroids")
    # print(descriptors)
    list_descriptors = []

    for descrs in descriptors:
       list_descriptors.extend(descrs.split())
       if ('' in list_descriptors):

            print("empty descriptor found in in list_descriptors")


    all_normalized_descriptors = [ds.strip().replace("*", "").lower() for ds in list_descriptors if len(ds) > 1]

    #all_normalized_descriptors = sorted(list(filter(''.__ne__, all_normalized_descriptors))) # removing empty '' words

    print("all_normalized_descriptors")
    print(all_normalized_descriptors)

    if (len (all_normalized_descriptors)) > 0:

        return max(set(all_normalized_descriptors), key=all_normalized_descriptors.count) # searching for mode

    else:qqq

        return 'empty_center'




def transform_labels_to_names(labels_ward_1, labels_ward_2, descriptors):
    labels_ward_1_named = []
    labels_ward_2_named = []

    # print(get_centroid_feature_for_cluster([descrs for descrs in descriptors[labels_ward_2 ==2] ]))

    for lbl in labels_ward_1:
        # print("label")
        # print(lbl)
        labels_ward_1_named.append(
            get_centroid_feature_for_cluster([descrs for descrs in descriptors[labels_ward_1 == lbl]]))

    for lbl in labels_ward_2:
        # print("label")
        # print(lbl)
        labels_ward_2_named.append(
            get_centroid_feature_for_cluster([descrs for descrs in descriptors[labels_ward_2 == lbl]]))

    return  labels_ward_1_named, labels_ward_2_named


def hierarchical_clustering(dict_from_csv,current_langugage):

    df_dict_from_csv = pd.DataFrame(dict_from_csv)
    descriptors = df_dict_from_csv[current_langugage+'hetmans']
    phrases =  df_dict_from_csv[current_langugage]
    df_dict_from_csv.fillna('null', inplace=True)

    # Calculate the linkage: mergings
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors).todense()
    mergings = linkage(X, method="complete")

    Z_ward = ward(X)


   # ClusterNode(Z[1])
    names = tfidf.get_feature_names()
    print("len(names)")
    print(len(names))

    cutree = cut_tree(mergings,  height=1.2)
    #cutree = cut_tree(Z_ward,  n_clusters=24)

    # print("cutree")
    # print(cutree)
    # print(cutree.shape)

    print ("height 1.2")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(mergings, height=1.3)
    #print ("height 1.3")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(mergings, height=1)
    #print("height 1")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(mergings, height=0)
    #print("height 0")
    #print_cut_tree_features(cutree, names)


    to_tree_obj = sch.to_tree(Z_ward)
    #print("to_tree_obj")
    #print(to_tree_obj)
    #print("pre_order length")
    #print(len(to_tree_obj.pre_order()))
    #print("to_tree_obj.get_count()")
    #print(to_tree_obj.get_count())

    #Y = ward(pdist(X))
    #Z = linkage(Y)
    #leaders = scipy.cluster.hierarchy.leaders(X, Z)



    y_dist = pdist(X)
    Z_centroid = centroid(y_dist)


    print("leaders")
    print(leaders)
    dendrogram(mergings,

               leaf_rotation=90,
               leaf_font_size=6,
               )
    plt.show()

    #labels = sch.fcluster(mergings, 1, criterion='distance')
    # print("mergings t=10 labels", labels)
    #
    # print("mergings shape")
    # print(mergings.shape)
    # print(mergings[0])
    # print(mergings[2])
    # print(mergings[248])
    # print(mergings[286])
    #
    # print(mergings[0][0])
    # print("Z_ward shape")
    # print(Z_ward.shape)
    # print("centroid shape")
    # print(Z_centroid.shape)

    # idxs = [33, 68, 62]
    # plt.figure(figsize=(10, 8))
    #
    # plt.scatter([X[:,0]],[X[:,1]])# plot all points
    # plt.scatter([X[idxs, 0]], [X[idxs, 1]], c='r')  # plot interesting points in red again
    # plt.show()

    cutree = cut_tree(Z_centroid, height=1)
    #print("height 1")
    #print_cut_tree_features(cutree, names)

    cutree = cut_tree(Z_centroid, height=3)
    #print("height 3")
    #print_cut_tree_features(cutree, names)

    labeled_rows_by_numbers_ward_level1 = fcluster(Z_ward, 2,
                             criterion='distance')  # LET'S ASSIGN NUMBER ONE TO THE UPPER LEVEL. HEIGT 2 IS HIGHER THAN 1 IN THE TREE


    labeled_rows_by_number_ward_level2 = fcluster(Z_ward, 1, criterion='distance')

    print("labeled_rows_by_number_ward_level2")
    print(labeled_rows_by_number_ward_level2)

    labels_ward_1_named, labels_ward_2_named = transform_labels_to_names(labeled_rows_by_numbers_ward_level1, labeled_rows_by_number_ward_level2, descriptors)

    print("labels_ward_1_named")
    print(labels_ward_1_named)
    print("labels_ward_2_named")
    print(labels_ward_2_named)
    pd.set_option('display.max_rows', None)  # TO AVOID TRUNCATING TABLE WHEN PRINTING
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1000)
    df_ward = pd.DataFrame({'phrases': phrases, 'labels2': labels_ward_2_named, 'labels1': labels_ward_1_named})

    df_ward_labels2_phrases = pd.DataFrame({'subdescriptors': labels_ward_2_named, 'phrases': phrases, "subclusters": labels_ward_2_named, "clusters": labels_ward_1_named }).groupby('subdescriptors').agg(', '.join)
    df_ward_labels2_phrases["subclusters"]=df_ward_labels2_phrases.index

    df_ward_labels1_labels2 = pd.DataFrame({"descriptors": labels_ward_1_named,  "clusters": labels_ward_1_named, "subclusters": labels_ward_2_named  }).sort_values(by=['descriptors']).drop_duplicates()


    df_ward_labels1_labels2 = df_ward_labels1_labels2.groupby('descriptors').agg(', '.join)
    df_ward_labels1_labels2["clusters"] = df_ward_labels1_labels2.index



    print("df_ward_labels1_labels2")
    print(df_ward_labels1_labels2)
    print("df_ward_labels2_phrases")
    print(df_ward_labels2_phrases)



    print("df_ward_grouped")
    print(df_ward.keys())

    print("df_ward.to_json()")
    print(df_ward_labels2_phrases.to_json(orient='records'))

    return df_ward_labels2_phrases.to_json(orient='records'), df_ward_labels1_labels2.to_json(orient='records')





def k_means_subclusters_phrases(csv_dict_structured_data, current_language, number_or_subclusters):

    rng = range(number_or_subclusters-1,number_or_subclusters)
    inertias = []

    df_structured_data = pd.DataFrame(csv_dict_structured_data) #converting ordered dict to dataframe
    descriptors = df_structured_data[current_language+'hetmans']

    df_structured_data.fillna('null', inplace=True) # replacing Nan cells with null for js to understand during json parsing

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors)

    for k in rng:
        model = KMeans(n_clusters=k, random_state=17)
        model.fit(X)
        inertias.append(model.inertia_)


    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    # print(len(terms))

    # phrase_numbers = model.cluster_centers_
    # print(phrase_numbers)
    #print(tfidf.decode(phrase_numbers[0]))

    labels = model.labels_  # the model assigns as many labels as rows are in the table
    print("k_means model.labels_ set")
    print(set(labels))
    print("model.labels_ LENGHT")
    print(len((model.labels_)))

    named_labels = [terms[order_centroids[label, :1][0]] for label in labels]

    # print("df_structured_data.keys()")
    # print(df_structured_data.keys())

    df_k_means_clusters = pd.DataFrame({
                                     "ukrlang": df_structured_data["ukrlang"],
                                                    "engllang": df_structured_data["engllang"],
                                                    "spanishlang": df_structured_data["spanishlang"],
                                                    "frenchlang": df_structured_data["frenchlang"],
                                                    "itallang": df_structured_data["itallang"],
                                                    "ruslang": df_structured_data["ruslang"],
                                                    "latinlang": df_structured_data["latinlang"],
                                                    "hebrlang": df_structured_data["hebrlang"],
                                        "hetmans": df_structured_data[current_language+'hetmans'],
                                         "subdescriptors": named_labels,
                                        "subclusters": named_labels
                                        }).groupby('subdescriptors').agg(','.join)

    df_k_means_clusters["subclusters"] = df_k_means_clusters.index




    # "centroid": current_centroid,
    # "subcentroid": current_neighbors,
    # "ukrlangphrases": df_structured_data["ukrlang"][row_number],
    # "spanlangphrases": df_structured_data["spanishlang"][row_number],
    # "engllangphrases": df_structured_data["engllang"][row_number],
    # "frenchlangphrases": df_structured_data["frenchlang"][row_number],
    # "itallangphrases": df_structured_data["itallang"][row_number],
    # "latinlangphrases": df_structured_data["latinlang"][row_number],
    # "ruslangphrases": df_structured_data["ruslang"][row_number],
    # "hebrlangphrases": df_structured_data["hebrlang"][row_number]

    pd.set_option('display.max_rows', None)  # TO AVOID TRUNCATING TABLE WHEN PRINTING
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1000)
    print("Current level df_k_means_cluster: ")
    print(df_k_means_clusters)
    print(df_k_means_clusters.index)
    return df_k_means_clusters


def k_means_subclusters_descriptors(csv_dict_structured_data, current_language, number_or_subclusters):
    rng = range(number_or_subclusters - 1, number_or_subclusters)
    inertias = []


    df_structured_data = pd.DataFrame(csv_dict_structured_data)  # converting ordered dict to dataframe
    descriptors = df_structured_data['hetmans']


    df_structured_data.fillna('null',
                              inplace=True)  # replacing Nan cells with null for js to understand during json parsing



    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors)

    for k in rng:
        model = KMeans(n_clusters=k, random_state=17)
        model.fit(X)
        inertias.append(model.inertia_)


    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()

    labels = model.labels_  # the model assigns as many labels as rows are in the table
    print("k_means model.labels_ set")
    print(set(labels))
    print("model.labels_ LENGHT")
    print(len((model.labels_)))

    named_labels = [terms[order_centroids[label, :1][0]] for label in labels]

    print("df_structured_data")
    print(df_structured_data)
    df_k_means_clusters = pd.DataFrame(
    {"subclusters": df_structured_data.index, #index, because in df_structured_data 'subclusters' is the name of rows, not column

    "descriptors": named_labels, "clusters": named_labels}).groupby('clusters').agg(', '.join)
    df_k_means_clusters['clusters'] = df_k_means_clusters.index

    # "centroid": current_centroid,
    # "subcentroid": current_neighbors,
    # "ukrlangphrases": df_structured_data["ukrlang"][row_number],
    # "spanlangphrases": df_structured_data["spanishlang"][row_number],
    # "engllangphrases": df_structured_data["engllang"][row_number],
    # "frenchlangphrases": df_structured_data["frenchlang"][row_number],
    # "itallangphrases": df_structured_data["itallang"][row_number],
    # "latinlangphrases": df_structured_data["latinlang"][row_number],
    # "ruslangphrases": df_structured_data["ruslang"][row_number],
    # "hebrlangphrases": df_structured_data["hebrlang"][row_number]

    pd.set_option('display.max_rows', None)  # TO AVOID TRUNCATING TABLE WHEN PRINTING
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1000)

    print("Current level df_k_means_cluster: ")
    print(df_k_means_clusters)
    print(df_k_means_clusters.index)

    return df_k_means_clusters.to_json(orient='records')


app = Flask(__name__)
@app.route('/')
def main():
    dict_from_csv = list(csv.DictReader(open('phrases.csv')))

    current_language = "ukrlang"



    df_kmeans_subdescriptors_phrases = k_means_subclusters_phrases(dict_from_csv, current_language,  48)
    k_means_descriptors_subdescriptors = k_means_subclusters_descriptors(df_kmeans_subdescriptors_phrases, current_language, 10)


    json_subdescriptors_phrases_hierarchical, json_descriptors_subdescriptors_hierarchical = hierarchical_clustering(dict_from_csv, current_language)
    return render_template('dict_output.html',
                           df_kmeans_subdescriptors_phrases=df_kmeans_subdescriptors_phrases.to_json(orient='records'), k_means_descriptors_subdescriptors=k_means_descriptors_subdescriptors,
                           json_subdescriptors_phrases_hierarchical=json_subdescriptors_phrases_hierarchical, json_descriptors_subdescriptors_hierarchical=json_descriptors_subdescriptors_hierarchical)

def draw_classifier_plot(rng, inertias):
    plt.plot(rng, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(rng)
    plt.show()


@app.route('/get_language', methods=['POST'])
def get_language():

    dict_from_csv = list(csv.DictReader(open('phrases.csv')))
    print("Changing descrpitpors language")
    current_language = request.form.get('select_language')

    if(current_language is None): current_language = "ukrlang"
    print(current_language)
    descriptors = [descr[current_language+'hetmans'] for descr in dict_from_csv]
    dict_centroid_phrases, labels = k_means_classifier(descriptors, dict_from_csv)

    subclusters_dict = get_k_means_subclusters_array(dict_from_csv, labels)
    dict_subcluster_phrases = k_means_subclusters_classifier(subclusters_dict, current_language)

    dict_centroid_phrases_hierarchical, labels2 = hierarchical_clustering(dict_from_csv, current_language)

    return render_template('dict_output.html',
                           dict_centroid_phrases=dict_subcluster_phrases,
                           dict_centroid_phrases_hierarchical=dict_centroid_phrases_hierarchical)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)









