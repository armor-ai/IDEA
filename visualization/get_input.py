# Preprocess the topic results to get the input for visualization
# -*- coding: utf-8 -*-
# __date__    ＝ "20/4/2017"

import os, sys

def save_input(fp_out, data):
	fw = open(fp_out, "w")
	fw.write("key,value,label,emerlabel,version,sent1,num\n")
	for idx, item in enumerate(data):
		fw.write(",".join(item)+"\n")
	print("Finish writing topic and labels to file.")

def get_num_label(fp_label, fp_num, fp_emerging, fp_label_sen, fp_emer_sen):
	topic_num   = open(fp_num).readlines()
	topic_label = open(fp_label).readlines()
	topic_emer  = open(fp_emerging).readlines()
	label_sen   = open(fp_label_sen).readlines()
	emer_sen    = open(fp_emer_sen).readlines()


	input_dict  = {}
	topic_dict  = {}
	versions = []
	for idx, raw in enumerate(topic_label):
		if raw.startswith("time"):
			version = raw.strip("\n").split()[-1]
			nums = topic_num[len(versions)].split("\t")
			versions.append(version)
			input_dict[version] = []
			continue
		labels = raw.split()[2:]
		num_label = raw.split()[1].split("\t")[0].split(":")
		number    = num_label[0]
		labels = [num_label[1]] + labels
		clean_labels = "; ".join([" ".join(l.split("_")) for l in labels if "_" in l])
		# process emerging topics, the first version does not have emerging topics
		emergs = topic_emer[idx-emeg_num].split()[2:]

		if emergs != ["None"]:
			clean_emers = "; ".join([" ".join(l.split("_")) for l in emergs if "_" in l])
		else:
			clean_emers = "0"
		
		# process topic sentences
		sents        = "".join(label_sen[idx].split(":")[1:]).strip("\n")
		sent_labels  = "".join([str(int(idx/2)+1)+ ": " +sent+";<br>" for idx, sent in enumerate(sents.split("\t")) if (idx%2 == 0)][:-1])

		# process emerging sentences
		try:
			emergs = "".join(emer_sen[idx-emeg_num].split(":")[1:]).strip(" \n")
			if emergs != "None":
				sent_labels  = "".join([str(int(idx/2)+1)+ ": " + sent+";<br>" for idx, sent in enumerate(emergs.split("\t")) if (idx%2 == 0)][:-1])
		except IndexError:
			pass

		input_dict[version].append(("topic" + number, nums[int(number)], clean_labels, clean_emers, version, sent_labels, str(len(versions))))

		if number not in topic_dict:
			topic_dict[number] = []
		topic_dict[number].append(("topic" + number, nums[int(number)], clean_labels, clean_emers, version, sent_labels, str(len(versions))))
	
	inputs = [item for sublist in input_dict.values() for item in sublist]
	topics = []
	for i in range(k):
		topics += topic_dict[str(i)]


	save_input(fp_out, topics)







if __name__ == "__main__":
	# for Youtube
	if len(sys.argv) < 3:
		print("Usage: python %s <result_dir> <K>" % sys.argv[0])
		print("\tresult_dir     the output dir of IDEA, should contain apk name, e.g., '../result_dir/youtube'")
		print("\tK     the number of topics")
		exit(1)

	result_dir = sys.argv[1]
	k = int(sys.argv[2])
	emeg_num = k+1
	fp_num = os.path.join(result_dir, "topic_width")
	fp_label = os.path.join(result_dir, "topic_labels")
	fp_emerging  = os.path.join(result_dir, "emerging_topic_labels")
	fp_label_sen = os.path.join(result_dir, "topic_sents")
	fp_emer_sen  = os.path.join(result_dir, "emerging_topic_sents")
	fp_out   = os.path.join("topic_label.csv")
	get_num_label(fp_label, fp_num, fp_emerging, fp_label_sen, fp_emer_sen)



