import re

def is_int(string):
    try:
        int(string)
        return True
    except:
        return False

def read_csv(path):
	data = open(path).readlines()
	#names = data[0].replace('\n', '').split('|')
	names = ['path','gloss']
	save_arr = []
	for line in data:
		save_dict = {name: 0 for name in names}
		line = line.replace('\n', '').split('|')
		for name, item in zip(names, line):
			save_dict[name] = item
		save_arr.append(save_dict)
	return save_arr

def is_float(string):
	try:
		float(string)
		return True
	except:
		return False

def make_gloss_dict(paths):
	data = []
	for path in paths:
		res = read_csv(path)
		data.extend(res)

	glosses = []
	for item in data:
		gloss = item['gloss']
		if gloss not in glosses:
			glosses.append(gloss)
	gloss_dict = {g:i for i,g in enumerate(glosses)}
	i2g_dict = {i:g for i,g in enumerate(glosses)}
	return gloss_dict, i2g_dict

def extract(message, gloss_dict):

	answers = message.split('\n')
	clean = []

	#try:
	for a in answers:
		if ':' not in a:
			floats = re.findall(r"[-+]?(?:\d*\.*\d+)", a)
			scores = []
			if len(floats) == 0:
				continue
			else:
				for f in floats:
					if (not is_int(f)) and (float(f) < 1):
						scores.append(f)

				if len(scores) == 0:
					continue
				if len(scores) == 1:
					score = float(scores[0])
				if len(scores) > 1:
					continue

			sent = a
			for f in floats:
				sent = sent.replace(f,'')
			sent = re.sub(r'[^\w\s]', '', sent)

		else:
			sent, score = a.split(':')
			sent = re.sub(r'[^\w\s]', '', sent)
			score = re.findall(r"[-+]?(?:\d*\.*\d+)", score)
			if len(score) == 0:
				continue
			score = score[0]
			try:
				score = float(score)
			except:
				print(a)
				continue

		c_sent = []
		for i, w in enumerate(sent.split()):
			if i==0 and is_float(w):
				continue
			if w in gloss_dict:
				c_sent.append(w)
		c_sent = ' '.join(c_sent)

		if len(c_sent) > 0:
			item = (c_sent, score)
			clean.append(item)
	# except:
	# 	return []

	return clean

prefix = '../GSL_isol'
train_gt_path = '../GSL_iso_files/sd/train_greek_iso.csv'
test_gt_path = '../GSL_iso_files/sd/test_greek_iso.csv'

gloss_dict, i2g_dict = make_gloss_dict([train_gt_path, test_gt_path])

text = '''1. ΠΑΡΑΚΑΛΩ (confidence score: 0.8553)
2. ΠΑΡΑΚΑΛΩ (confidence score: 0.8515)
3. ΠΑΡΑΚΑΛΩ (confidence score: 0.8471)
4. ΠΑΡΑΚΑΛΩ (confidence score: 0.8378)
5. ΠΑΡΑΚΑΛΩ (confidence score: 0.8068)

The given sentence only contains repetitions of the same sign gloss "ΠΑΡΑΚΑΛΩ", which means "please" in Greek Sign Language. Since the repeated glosses need to be removed and the order of the words should not be changed, these are the 5 possible choices for an edited sentence.

Note: Since the edited sentence contains only one sign gloss, there aren't 10 possible choices.'''

text = '''1. ΕΘΝΙΚΗ ΖΑΛΙΖΟΜΑΙ ΕΠΙΔΟΜΑ (0.90)
2. ΕΘΝΙΚΗ ΕΠΙΔΟΜΑ (0.85)
3. ΖΑΛΙΖΟΜΑΙ ΕΠΙΔΟΜΑ (0.80)
4. ΕΘΝΙΚΗ ΖΑΛΙΖΟΜΑΙ (0.75)
5. ΕΘΝΙΚΗ ΕΠΙΔΟΜΑ ΕΠΙΔΟΜΑ (0.70)
6. ΕΘΝΙΚΗ ΖΑΛΙΖΟΜΑΙ ΕΠΙΔΟΜΑ Ε'''

clean = extract(text, gloss_dict)
print(clean)







