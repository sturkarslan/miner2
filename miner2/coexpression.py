import datetime,numpy,pandas,time,sys,itertools
import sklearn,sklearn.decomposition
import multiprocessing, multiprocessing.pool
from collections import Counter

# Some default constants if the user does not specify any
# default number of iterations for algorithms with iterations
NUM_ITERATIONS = 25
MIN_NUM_GENES = 6
MIN_NUM_OVEREXP_SAMPLES = 4
MAX_SAMPLES_EXCLUDED = 0.5
RANDOM_STATES = 12
OVEREXP_THRESHOLD = 80
NUM_CORES = 1
RECONSTRUCTION_THRESHOLD = 0.925
SIZE_LONG_SET = 50


def cluster(expressionData, min_number_genes=6, min_number_overexp_samples=4,
            max_samples_excluded=0.50, random_state=12, overexpression_threshold=80,
            pct_threshold=80):


    df = expressionData.copy()
    maxStep = int(numpy.round(10 * max_samples_excluded))
    allGenesMapped = []
    bestHits = []

    zero = numpy.percentile(expressionData,0)
    expressionThreshold = numpy.mean([numpy.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero], overexpression_threshold) for i in range(expressionData.shape[1])])

    startTimer = time.time()
    trial = -1
    for step in range(maxStep):
        trial+=1
        progress = (100./maxStep)*trial
        print('{:.2f} percent complete'.format(progress))
        genesMapped = []
        bestMapped = []

        pca = sklearn.decomposition.PCA(10,random_state=random_state)
        principalComponents = pca.fit_transform(df.T)
        principalDf = pandas.DataFrame(principalComponents)
        principalDf.index = df.columns

        for i in range(10):
            pearson = pearson_array(numpy.array(df), numpy.array(principalDf[i]))
            if len(pearson) == 0:
                continue
            highpass = max(numpy.percentile(pearson,95), 0.1)
            lowpass = min(numpy.percentile(pearson,5), -0.1)
            cluster1 = numpy.array(df.index[numpy.where(pearson > highpass)[0]])
            cluster2 = numpy.array(df.index[numpy.where(pearson < lowpass)[0]])

            for clst in [cluster1,cluster2]:
                pdc = recursive_alignment(clst, expression_data=df,
                                          min_number_genes=min_number_genes,
                                          pct_threshold=pct_threshold)
                if len(pdc)==0:
                    continue
                elif len(pdc) == 1:
                    genesMapped.append(pdc[0])
                elif len(pdc) > 1:
                    for j in range(len(pdc)-1):
                        if len(pdc[j]) > min_number_genes:
                            genesMapped.append(pdc[j])

        allGenesMapped.extend(genesMapped)
        try:
            stackGenes = numpy.hstack(genesMapped)
        except:
            stackGenes = []
        residualGenes = list(set(df.index)-set(stackGenes))
        df = df.loc[residualGenes,:]

        # computationally fast surrogate for passing the overexpressed significance test:
        for ix in range(len(genesMapped)):
            tmpCluster = expressionData.loc[genesMapped[ix],:]
            tmpCluster[tmpCluster<expressionThreshold] = 0
            tmpCluster[tmpCluster>0] = 1
            sumCluster = numpy.array(numpy.sum(tmpCluster, axis=0))
            numHits = numpy.where(sumCluster > 0.333 * len(genesMapped[ix]))[0]
            bestMapped.append(numHits)
            if len(numHits) > min_number_overexp_samples:
                bestHits.append(genesMapped[ix])

        if len(bestMapped) > 0:
            countHits = Counter(numpy.hstack(bestMapped))
            ranked = countHits.most_common()
            dominant = [i[0] for i in ranked[0:int(numpy.ceil(0.1*len(ranked)))]]
            remainder = [i for i in numpy.arange(df.shape[1]) if i not in dominant]
            df = df.iloc[:,remainder]

    bestHits.sort(key=lambda s: -len(s))

    stopTimer = time.time()
    print('\ncoexpression clustering completed in {:.2f} minutes'.format((stopTimer-startTimer)/60.))
    return bestHits


def combine_clusters(axes, clusters, threshold):
    combine_axes = {}
    filter_keys = numpy.array(list(axes.keys())) # ALO: changed to list because of Py3
    axes_matrix = numpy.vstack([axes[i] for i in filter_keys])
    for key in filter_keys:
        axis = axes[key]
        pearson = pearson_array(axes_matrix,axis)
        combine = numpy.where(pearson > threshold)[0]
        combine_axes[key] = filter_keys[combine]

    revised_clusters = {}
    combined_keys = decompose_dictionary_to_lists(combine_axes)
    for key_list in combined_keys:
        genes = list(set(numpy.hstack([clusters[i] for i in key_list])))
        revised_clusters[len(revised_clusters)] = sorted(genes)

    return revised_clusters


def decompose(geneset, expressionData, minNumberGenes=6, pct_threshold=80):
    fm = make_frequency_matrix(expressionData.loc[geneset,:])
    tst = numpy.multiply(fm,fm.T)
    tst[tst < numpy.percentile(tst,pct_threshold)]=0
    tst[tst > 0]=1
    unmix_tst = unmix(tst)
    unmixedFiltered = [i for i in unmix_tst if len(i)>=minNumberGenes]
    return unmixedFiltered


def decompose_dictionary_to_lists(dict_):
    decomposedSets = []
    for key in list(dict_.keys()):
        newSet = iterative_combination(dict_, key, iterations=NUM_ITERATIONS)
        if newSet not in decomposedSets:
            decomposedSets.append(newSet)
    return decomposedSets


def make_frequency_matrix(matrix, overExpThreshold = 1):

    final_index = None
    if type(matrix) == pandas.core.frame.DataFrame:
        final_index = matrix.index
        matrix = numpy.array(matrix)

    index = numpy.arange(matrix.shape[0])
    matrix[matrix<overExpThreshold] = 0
    matrix[matrix>0] = 1
    frequency_dictionary = {name:[] for name in index}

    for column in range(matrix.shape[1]):
        hits = numpy.where(matrix[:,column]>0)[0]
        geneset = index[hits]
        for name in geneset:
            frequency_dictionary[name].extend(geneset)

    fm = numpy.zeros((len(index),len(index)))
    for key in list(frequency_dictionary.keys()):
        tmp = frequency_dictionary[key]
        if len(tmp) == 0:
            continue
        count = Counter(tmp)
        results_ = numpy.vstack(list(count.items()))
        fm[key,results_[:,0]] = results_[:,1]/float(count[key])

    fm_df = pandas.DataFrame(fm)

    if final_index is not None:
        fm_df.index = final_index
        fm_df.columns = final_index

    return fm_df



def get_axes(clusters, expression_data):
    axes = {}
    for key in clusters.keys():
        genes = clusters[key]
        fpc = sklearn.decomposition.PCA(1)
        principal_components = fpc.fit_transform(expression_data.loc[genes,:].T)
        axes[key] = principal_components.ravel()
    return axes


def gene_mapper(task):
    genes_mapped = []
    df, principal_df, i, min_number_genes = task
    pearson = pearson_array(numpy.array(df), numpy.array(principal_df[i]))
    highpass = max(numpy.percentile(pearson,95), 0.1)
    lowpass = min(numpy.percentile(pearson,5), -0.1)
    cluster1 = numpy.array(df.index[numpy.where(pearson > highpass)[0]])
    cluster2 = numpy.array(df.index[numpy.where(pearson < lowpass)[0]])

    for clst in [cluster1, cluster2]:
        pdc = recursive_alignment(clst, df, min_number_genes)
        if len(pdc) == 0:
            continue
        elif len(pdc) == 1:
            genes_mapped.append(pdc[0])
        elif len(pdc) > 1:
            for j in range(len(pdc)-1):
                if len(pdc[j]) > min_number_genes:
                    genes_mapped.append(pdc[j])

    return genes_mapped


def iterative_combination(dict_, key, iterations=25):
    initial = dict_[key]
    initialLength = len(initial)
    for iteration in range(iterations):
        revised = [i for i in initial]
        for element in initial:
            revised = list(set(revised)|set(dict_[element]))
        revisedLength = len(revised)
        if revisedLength == initialLength:
            return revised
        elif revisedLength > initialLength:
            initial = [i for i in revised]
            initialLength = len(initial)
    return revised



def make_hits_matrix_new(matrix): ### new function developed by Wei-Ju
    num_rows = matrix.shape[0]
    hits_values = numpy.zeros((num_rows,num_rows))

    for column in range(matrix.shape[1]):
        geneset = matrix[:,column]
        hits = numpy.where(geneset > 0)[0]
        rows = []
        cols = []
        cp = itertools.product(hits, hits)
        for row, col in cp:
            rows.append(row)
            cols.append(col)
        hits_values[rows, cols] += 1

    return hits_values


def parallel_overexpress_surrogate(task):
    element, expression_data, expression_threshold = task

    tmp_cluster = expression_data.loc[element, :]
    tmp_cluster[tmp_cluster < expression_threshold] = 0
    tmp_cluster[tmp_cluster > 0] = 1
    sum_cluster = numpy.array(numpy.sum(tmp_cluster, axis=0))
    hits = numpy.where(sum_cluster > 0.333 * len(element))[0]

    return (element, hits)


def pearson_array(array, vector):
    ybar = numpy.mean(vector)
    sy = numpy.std(vector, ddof=1)
    yterms = (vector - ybar) / float(sy)

    array_sx = numpy.std(array, axis=1, ddof=1)

    if 0 in array_sx:
        pass_index = numpy.where(array_sx > 0)[0]
        array = array[pass_index, :]
        array_sx = array_sx[pass_index]

    array_xbar = numpy.mean(array, axis=1)
    product_array = numpy.zeros(array.shape)

    for i in range(0,product_array.shape[1]):
        product_array[:, i] = yterms[i] * (array[:, i] - array_xbar) / array_sx

    return numpy.sum(product_array, axis=1) / float(product_array.shape[1]-1)


def process_coexpression_lists(lists,expressionData,threshold=0.925):
    reconstructed = reconstruction(lists,expressionData,threshold)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys()]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList


def reconstruction(decomposed_list, expression_data, threshold=RECONSTRUCTION_THRESHOLD):
    clusters = {i:decomposed_list[i] for i in range(len(decomposed_list))}
    axes = get_axes(clusters, expression_data)
    return combine_clusters(axes, clusters, threshold)


def recursive_alignment(geneset, expression_data, min_number_genes=6, pct_threshold=80):
    recDecomp = recursive_decomposition(geneset, expression_data, min_number_genes, pct_threshold)
    if len(recDecomp) == 0:
        return []

    reconstructed = reconstruction(recDecomp,expression_data)
    reconstructedList = [reconstructed[i] for i in list(reconstructed.keys())
                         if len(reconstructed[i]) > min_number_genes]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList


def recursive_decomposition(geneset, expressionData, minNumberGenes=6, pct_threshold=80):
    unmixedFiltered = decompose(geneset,expressionData,minNumberGenes,pct_threshold)
    if len(unmixedFiltered) == 0:
        return []
    shortSets = [i for i in unmixedFiltered if len(i)<50]
    longSets = [i for i in unmixedFiltered if len(i)>=50]
    if len(longSets)==0:
        return unmixedFiltered
    for ls in longSets:
        unmixedFiltered = decompose(ls,expressionData,minNumberGenes,pct_threshold)
        if len(unmixedFiltered)==0:
            continue
        shortSets.extend(unmixedFiltered)
    return shortSets


def revise_initial_clusters(clusterList, expressionData, threshold=RECONSTRUCTION_THRESHOLD):
    coexpressionLists = process_coexpression_lists(clusterList,expressionData,threshold)
    coexpressionLists.sort(key= lambda s: -len(s))

    for iteration in range(5):
        previousLength = len(coexpressionLists)
        coexpressionLists = process_coexpression_lists(coexpressionLists,expressionData,threshold)
        newLength = len(coexpressionLists)
        if newLength == previousLength:
            break

    coexpressionLists.sort(key= lambda s: -len(s))
    coexpressionDict = {str(i):list(coexpressionLists[i]) for i in range(len(coexpressionLists))}

    return coexpressionDict


def unmix(df, iterations=NUM_ITERATIONS, return_all=False):    
    frequencyClusters = []

    for iteration in range(iterations):
        sumDf1 = df.sum(axis=1)
        maxSum = df.index[numpy.argmax(numpy.array(sumDf1))]
        hits = numpy.where(df.loc[maxSum] > 0)[0]
        hitIndex = list(df.index[hits])
        block = df.loc[hitIndex,hitIndex]
        blockSum = block.sum(axis=1)
        coreBlock = list(blockSum.index[numpy.where(blockSum>=numpy.median(blockSum))[0]])
        remainder = list(set(df.index)-set(coreBlock))
        frequencyClusters.append(coreBlock)
        if len(remainder)==0:
            return frequencyClusters
        if len(coreBlock)==1:
            return frequencyClusters
        df = df.loc[remainder,remainder]
    if return_all:
        frequencyClusters.append(remainder)
    return frequencyClusters


def remix(df,frequencyClusters):
    finalClusters = []
    for cluster in frequencyClusters:
        sliceDf = df.loc[cluster,:]
        sumSlice = sliceDf.sum(axis=0)
        cut = min(0.8,numpy.percentile(sumSlice.loc[cluster]/float(len(cluster)),90))
        minGenes = max(4,cut*len(cluster))
        keepers = list(sliceDf.columns[numpy.where(sumSlice>=minGenes)[0]])
        keepers = list(set(keepers)|set(cluster))
        finalClusters.append(keepers)
        finalClusters.sort(key = lambda s: -len(s))
    return finalClusters
