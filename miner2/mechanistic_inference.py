import datetime, pandas, numpy, os, pickle, sys
import sklearn, sklearn.decomposition
import scipy, scipy.stats
import multiprocessing
from pkg_resources import Requirement, resource_filename
import logging
import json
from collections import defaultdict

from miner2 import util, coexpression


def axis_tfs(axesDf, tfList, expressionData, correlation_threshold=0.3):
    axesArray = numpy.array(axesDf.T)
    if correlation_threshold > 0:
        tfArray = numpy.array(expressionData.loc[tfList,:])
    axes = numpy.array(axesDf.columns)
    tfDict = {}

    if type(tfList) is list:
        tfs = numpy.array(tfList)
    elif type(tfList) is not list:
        tfs = np.array(list(tfList))

    if correlation_threshold == 0:
        for axis in range(axesArray.shape[0]):
            tfDict[axes[axis]] = tfs
        return tfDict

    for axis in range(axesArray.shape[0]):
        tfDict_key = axes[axis]
        tfCorrelation = coexpression.pearson_array(tfArray,axesArray[axis,:])
        tfDict[tfDict_key] = tfs[numpy.where(numpy.abs(tfCorrelation) >= correlation_threshold)[0]]
    return tfDict



def enrichment(axes, revised_clusters, expression_data, correlation_threshold=0.3,
               num_cores=1, p=0.05,
               database="tfbsdb_tf_to_genes.pkl",
               database_path=None):

    logging.info("mechanistic inference")

    if database_path is None:
        tf_2_genes_path = resource_filename(Requirement.parse("miner2"),
                                            'miner2/data/{}'.format(database))
    else:
        tf_2_genes_path = database_path

    with open(tf_2_genes_path, 'rb') as f:
        tfToGenes = pickle.load(f)

    if correlation_threshold <= 0:
        allGenes = [int(len(expression_data.index))]
    elif correlation_threshold > 0:
        allGenes = list(expression_data.index)

    tfs = list(tfToGenes.keys())
    tfMap = axis_tfs(axes, tfs, expression_data, correlation_threshold=correlation_threshold)
    taskSplit = util.split_for_multiprocessing(list(revised_clusters.keys()), num_cores)
    tasks = [[taskSplit[i], (allGenes, revised_clusters, tfMap, tfToGenes, p)]
             for i in range(len(taskSplit))]
    tfbsdbOutput = multiprocess(tfbsdb_enrichment, tasks)
    return condense_output(tfbsdbOutput)


def multiprocess(function,tasks):
    import multiprocessing, multiprocessing.pool
    hydra=multiprocessing.pool.Pool(len(tasks))  
    output=hydra.map(function,tasks)   
    hydra.close()
    hydra.join()
    return output


def hyper(population,set1,set2,overlap):

    b = max(set1,set2)
    c = min(set1,set2)
    hyp = scipy.stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])

    return prb


def get_principal_df(dict_, expressionData, regulons=None, subkey='genes',
                     min_number_genes=8, random_state=12):
    pcDfs = []
    setIndex = set(expressionData.index)

    if regulons is not None:
        dict_, df = regulonDictionary(regulons)
    for i in list(dict_.keys()):
        if subkey is not None:
            genes = list(set(dict_[i][subkey])&setIndex)
            if len(genes) < min_number_genes:
                continue
        elif subkey is None:
            genes = list(set(dict_[i])&setIndex)
            if len(genes) < min_number_genes:
                continue

        pca = sklearn.decomposition.PCA(1, random_state=random_state)
        principalComponents = pca.fit_transform(expressionData.loc[genes,:].T)
        principalDf = pandas.DataFrame(principalComponents)
        principalDf.index = expressionData.columns
        principalDf.columns = [str(i)]

        normPC = numpy.linalg.norm(numpy.array(principalDf.iloc[:,0]))
        pearson = scipy.stats.pearsonr(principalDf.iloc[:,0],
                                       numpy.median(expressionData.loc[genes,:], axis=0))
        signCorrection = pearson[0] / numpy.abs(pearson[0])
        principalDf = signCorrection * principalDf / normPC

        pcDfs.append(principalDf)

    principalMatrix = pandas.concat(pcDfs, axis=1)
    return principalMatrix


def get_regulon_dictionary(regulons):
    regulonModules = {}
    df_list = []

    for tf in list(regulons.keys()):
        for key in list(regulons[tf].keys()):
            genes = regulons[tf][key]
            id_ = str(len(regulonModules))
            regulonModules[id_] = regulons[tf][key]
            for gene in genes:
                df_list.append([id_,tf,gene])

    array = numpy.vstack(df_list)
    df = pandas.DataFrame(array)
    df.columns = ["Regulon_ID", "Regulator", "Gene"]

    return regulonModules, df


def condense_output(output,output_type=dict):
    if output_type is dict:
        results = {}
        for i in range(len(output)):
            resultsDict = output[i]
            keys = list(resultsDict.keys())
            for j in range(len(resultsDict)):
                key = keys[j]
                results[key] = resultsDict[key]
        return results
    elif output_type is not dict:
        results = pandas.concat(output, axis=0)

    return results


def tfbsdb_enrichment(task):
    start, stop = task[0]
    allGenes, revisedClusters, tfMap, tfToGenes, p = task[1]
    keys = list(revisedClusters.keys())[start:stop]

    if len(allGenes) == 1:

        population_size = int(allGenes[0])
        clusterTfs = {}
        for key in keys:
            for tf in tfMap[str(key)]:
                hits0TfTargets = tfToGenes[tf]
                hits0clusterGenes = revisedClusters[key]
                overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
                if len(overlapCluster) <= 1:
                    continue
                pHyper = hyper(population_size,len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
                if pHyper < p:
                    if key not in list(clusterTfs.keys()):
                        clusterTfs[key] = {}
                    clusterTfs[key][tf] = [pHyper,overlapCluster]

    elif len(allGenes) > 1:
        population_size = len(allGenes)
        clusterTfs = {}
        for key in keys:
            for tf in tfMap[str(key)]:
                hits0TfTargets = list(set(tfToGenes[tf])&set(allGenes))
                hits0clusterGenes = revisedClusters[key]
                overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
                if len(overlapCluster) <= 1:
                    continue
                pHyper = hyper(population_size,len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
                if pHyper < p:
                    if key not in list(clusterTfs.keys()):
                        clusterTfs[key] = {}
                    clusterTfs[key][tf] = [pHyper,overlapCluster]

    return clusterTfs


def get_coregulation_modules(mechanistic_output):
    coregulation_modules = {}
    for i in mechanistic_output.keys():
        for key in mechanistic_output[i].keys():
            if key not in coregulation_modules.keys():
                coregulation_modules[key] = {}
            genes = mechanistic_output[i][key][1]
            coregulation_modules[key][i] = genes
    return coregulation_modules


def get_regulons(coregulationModules, min_number_genes=5, freq_threshold=0.333):
    regulons = {}
    keys = list(coregulationModules.keys())
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidence_matrix(coregulationModules, key=i, freq_threshold=freq_threshold)
        unmixed = coexpression.unmix(normDf)
        remixed = coexpression.remix(normDf, unmixed)
        if len(remixed) > 0:
            for cluster in remixed:
                if len(cluster) >= min_number_genes:
                    if tf not in list(regulons.keys()):
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster
    return regulons


def coincidence_matrix(coregulationModules, key, freq_threshold=0.333):
    tf = list(coregulationModules.keys())[key]
    subRegulons = coregulationModules[tf]
    srGenes = list(set(numpy.hstack([subRegulons[i] for i in subRegulons.keys()])))

    template = pandas.DataFrame(numpy.zeros((len(srGenes), len(srGenes))))
    template.index = srGenes
    template.columns = srGenes

    for key in list(subRegulons.keys()):
        genes = subRegulons[key]
        template.loc[genes,genes]+=1
    trace = numpy.array([template.iloc[i,i] for i in range(template.shape[0])]).astype(float)
    normDf = ((template.T)/trace).T
    normDf[normDf < freq_threshold] = 0
    normDf[normDf > 0] = 1

    return normDf


def get_coexpression_modules(mechanistic_output):
    coexpressionModules = {}
    for i in mechanistic_output.keys():
        genes = list(set(numpy.hstack([mechanistic_output[i][key][1]
                                       for key in mechanistic_output[i].keys()])))
        coexpressionModules[i] = genes
    return coexpressionModules


"""
Postprocessing functions
"""
def convert_dictionary(dic, conversion_table):
    conv_dict = defaultdict(list)
    for pref_name, name in conversion_table.iteritems():
        conv_dict[pref_name].append(name)

    converted = defaultdict(list)
    for i in dic.keys():
        genes = dic[i]
        for gene in genes:
            converted[i].extend(conv_dict[gene])
    return converted


def convert_regulons(df, conversion_table):
    """df is dataframe with the columns "Regulon_ID", "Regulator", "Gene"
    """
    reg_ids = []
    regs = []
    genes = []
    conv_dict = defaultdict(list)
    for pref_name, name in conversion_table.iteritems():
        conv_dict[pref_name].append(name)

    for i, row in df.iterrows():
        reg_ids.append(row["Regulon_ID"])
        regs.append(conv_dict[row["Regulator"]][0])
        genes.append(conv_dict[row["Gene"]][0])

    regulon_df_converted = pandas.DataFrame(numpy.vstack([reg_ids, regs, genes]).T)
    regulon_df_converted.columns = ["Regulon_ID","Regulator","Gene"]
    return regulon_df_converted
