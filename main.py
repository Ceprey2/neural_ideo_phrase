import csv
import json

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request
from flask import render_template
from scipy.cluster.hierarchy import fcluster, leaders, centroid
from scipy.cluster.hierarchy import linkage, dendrogram, ward, cut_tree
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

#print(ukr_descriptors[0:33])




#print(features_descriptors)
wcss=[]
number_of_clusters = 48
rng = range(49,50)
inertias = []

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

    all_normalized_descriptors = [ds.replace("*", "").strip().lower() for ds in list_descriptors]

    if (len (all_normalized_descriptors)) > 0:

        return max(set(all_normalized_descriptors), key=all_normalized_descriptors.count) # searching for mode

    else:
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


def hierarchical_clustering(descriptors, phrases, csv_structured_data):
    dict_centroid_phrases_hierarchical = []

    csv_structured_data.fillna('null', inplace=True)

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

    from scipy.cluster import hierarchy as sch

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

    from scipy.spatial.distance import pdist

    y_dist = pdist(X)
    Z_centroid = centroid(y_dist)


    print("leaders")
    print(leaders)
    dendrogram(mergings,

               leaf_rotation=90,
               leaf_font_size=6,
               )
    plt.show()

    labels = sch.fcluster(mergings, 1, criterion='distance')
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

    labels_ward_1 = fcluster(Z_ward, 1, criterion='distance')
    #print("ward labels 1")



    labels_ward_2 = fcluster(Z_ward, 2, criterion='distance')
    #print("ward labels 2")


    labels_ward_1_named, labels_ward_2_named = transform_labels_to_names(labels_ward_1, labels_ward_2, descriptors)

    pd.set_option('display.max_rows', None)  # TO AVOID TRUNCATING TABLE WHEN PRINTING
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1000)
    df_ward = pd.DataFrame({'phrases': phrases, 'labels2': labels_ward_2_named, 'labels1': labels_ward_1_named})
    #df_ward = df_ward.sort_values(by='labels1', ascending=False, na_position='first')

    print("DataFrame of two level labels")
    print(df_ward)




    # Create a DataFrame with labels and varieties as columns: df
    # df = pd.DataFrame({'labels': labels, 'descriptors': descriptors})


    # print("DataFrame of classifyier")
    # print(df[:20])
    # Create crosstab: ct
    #ct = pd.crosstab(df['labels'], df['descriptors'])

    # Display ct
    #print(ct)

    for lb1, lb2, ukrlangphrases in zip(labels_ward_1_named, labels_ward_2_named, csv_structured_data["ukrlang"]):
        entry = {
            "centroid": lb1,
            "subcentroid": lb2,
            "ukrlangphrases": ukrlangphrases
        }


        dict_centroid_phrases_hierarchical.append(json.dumps(entry))



    return dict_centroid_phrases_hierarchical, labels


def agglomerative_clustering(descriptors):
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors).todense()
    from sklearn.cluster import AgglomerativeClustering


    clustering = AgglomerativeClustering().fit(X)

    agglom_labels = clustering.labels_

    # print("agglom_labels")
    # print(agglom_labels)
    # print("agglom_labels length")
    # print(len(agglom_labels))



def k_means_classifier(descriptors, phrases, csv_structured_data):

    # print("type(csv_structured_data)")
    # print(type(csv_structured_data))
    df_structured_data = pd.DataFrame(csv_structured_data)
    df_structured_data.fillna('null', inplace=True)

    all_centroids = []

    entries = []
    dict_centroid_phrases = {"main_centroid": "", "entries": entries }
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+'?\w+\b")
    X = tfidf.fit_transform(descriptors)
    # print("X tipe of vectorizer")
    # print(type(X))
    # print("X.shape of transformed data")
    # print(X.shape)
    # print("Elements 1 transformed")
    # print(X[1])
    # print("Elements 5 transformed")
    # print(X[6])
    for k in rng:
        model = KMeans(n_clusters=k, random_state=17)
        model.fit(X)
        inertias.append(model.inertia_)

    try:
        draw_classifier_plot(rng, inertias)
    except:
        print('Plotting error')

    # plt.scatter(X,y_kmeans, c=y_kmeans, s=50, cmap='viridis')
    # wcss.append(model.inertia_)
    # plt.plot(range(1, 1), wcss)
    # plt.title('The Elbow Method Graph')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    # print(model.labels_)
    # print(len(model.labels_))

    # print(X.shape)
    # print(len(features_descriptors))
    # print("model.verbose")
    # print(model.verbose)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    # print(len(terms))

    # phrase_numbers = model.cluster_centers_
    # print(phrase_numbers)
    #print(tfidf.decode(phrase_numbers[0]))

    labels = model.labels_  # the model assigns as many labels as rows are in the table
    # print("model.labels_")
    # print(set(model.labels_))
    # print("model.labels_ LENGHT")
    # print(len(set(model.labels_)))
    for i in range(len(set(labels))):
        current_neighbors = []
        #print("Synonymicgroup %d:" % i),

        current_centroid = terms[order_centroids[i, :1][0]]
        all_centroids.append(current_centroid)
        print("Cluster %d:" % i),
        datapoinst_to_print = 25
        for ind in order_centroids[i, :datapoinst_to_print]:
            current_neighbors.append(terms[ind])
            # print(terms[ind])

        # print("current_centroid")  # searching for centroids amongst descriptors
        # print(current_centroid)  # searching for centroids amongst descriptors
        for k in range(len(csv_structured_data)):
            if ((i == labels[k])):
                # print("df_structured_data[\"ukrlang\"]")
                #print(df_structured_data["ukrlang"])
                print("k =", k)
                #print(df_structured_data[k])
                #entries_to_output.append(df_structured_data[k])
                entry = {
                    "centroid": current_centroid,
                    "subcentroid": current_neighbors,
                    "ukrlangphrases": df_structured_data["ukrlang"][k],
                    "spanlangphrases": df_structured_data["spanishlang"][k],
                    "engllangphrases": df_structured_data["engllang"][k],
                    "frenchlangphrases": df_structured_data["frenchlang"][k],
                    "itallangphrases": df_structured_data["itallang"][k],
                    "latinlangphrases": df_structured_data["latinlang"][k],
                    "ruslangphrases": df_structured_data["ruslang"][k],
                    "hebrlangphrases": df_structured_data["hebrlang"][k]

                }

                entries.append(json.dumps(entry))

        #print("current neighbors")
        #print(current_neighbors)

    dict_centroid_phrases["main_centroid"] = max(set(all_centroids), key=all_centroids.count)
    dict_centroid_phrases["entries"] = entries
    return dict_centroid_phrases, labels


def get_k_means_subclusters_array (csv_structured_data, labels):
    print("Lengthes")
    print (len(labels))
    print (len(csv_structured_data))

    clusters_list = []
    for i in set(labels):
        subcluster = [entry for entry, labl in zip(csv_structured_data, labels) if labl == i]
        # print(i, "label" )
        # print("subcluster")
        # print(subcluster)
        clusters_list.append(subcluster)



    return  clusters_list


def k_means_subclusters_classifier (df_clusters_list, phrases):


    dict_clusters_array = []
    #print("dataframe", "first element from dataframe")
    #print(df_clusters_list)
    #print(df_clusters_list[0])
    descriptors = []
    for cluster in df_clusters_list:
        descrs = [entry['ukrlanghetmans'] for entry in cluster]
        descriptors.extend(descrs)

    for df_cluster in df_clusters_list:
        #print("next subcluster length", len(df_cluster))
        dict_subcluster, labels = k_means_classifier(descriptors, phrases, df_cluster)
        #print(dict_subcluster)
        dict_clusters_array.append(json.dumps(dict_subcluster))

    return dict_clusters_array





app = Flask(__name__)
@app.route('/')
def main():
    csv_structured_data = pd.read_csv('all_phrases.csv')
    #print("len(csv_structured_data)")
    #print(len(csv_structured_data))
    descriptors = csv_structured_data['ukrlanghetmans']
    ukrphrases = csv_structured_data['ukrlang']
    dict_from_csv = list(csv.DictReader(open('all_phrases.csv')))

    dict_centroid_phrases, labels = k_means_classifier(descriptors, ukrphrases, dict_from_csv)
    #print("labels from k_means")
    #print(labels)

    subclusters_dict = get_k_means_subclusters_array(dict_from_csv, labels)
    dict_subcluster_phrases = k_means_subclusters_classifier(subclusters_dict, ukrphrases)
    print("subclusters dict")
    print(dict_subcluster_phrases)
    print("dict_centroid_phrases")
    print(dict_centroid_phrases)
    dict_centroid_phrases_hierarchical, labels2 = hierarchical_clustering(descriptors, ukrphrases, csv_structured_data)
    agglomerative_clustering(descriptors)

    #print('all_centroids', all_centroids)

    #get_select_language_option()
    #print("dict_centroid_phrases_hierarchical")
    #print(type(dict_centroid_phrases_hierarchical))
    #print("dict_centroid_phrases")
    #print(type(dict_centroid_phrases))
    return render_template('dict_output.html', dict_centroid_phrases=dict_centroid_phrases, dict_centroid_phrases_hierarchical=dict_centroid_phrases_hierarchical)
    # return "OK"



@app.route('/get_language', methods=['POST'])
def get_language():
        csv_raw_data = pd.read_csv('all_phrases.csv', dtype={'ukrlanghetmans': str}, )
        print('Function get descriptors selected language started')
        if request.method == 'POST':
            print("Changing descrpitpors language")
            current_language = request.form.get('select_language')
            print(current_language)

            descriptors = csv_raw_data[current_language+"hetmans"]
            phrases = csv_raw_data[current_language]
            dict_centroid_phrases, labels = k_means_classifier(descriptors, phrases)
            dict_centroid_phrases_hierarchical, labels2 = hierarchical_clustering(descriptors, phrases)


            return render_template('dict_output.html', dict_centroid_phrases=dict_centroid_phrases, dict_centroid_phrases_hierarchical=dict_centroid_phrases_hierarchical)

        else:
            return "NO OK"



def draw_classifier_plot(rng, inertias):
    plt.plot(rng, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(rng)
    plt.show()



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)









