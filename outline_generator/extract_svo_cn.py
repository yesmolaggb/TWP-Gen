import spacy
from collections.abc import Iterable

# Load Chinese language model
# nlp = spacy.load('en_core_web_trf')
nlp = spacy.load('zh_core_web_lg')

# dependency markers for subjects in Chinese
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}
# dependency markers for objects in Chinese
OBJECTS = {"dobj", "iobj", "attr", "pobj"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB", "PUNCT"}
# POS tags that work as NOUN in Chinese
NOUN_POS = {"PROPN", "NOUN", "PRON"}
# words that are negations in Chinese
NEGATIONS = {"不", "没", "没有", "无", "非", "莫", "勿", "未", "否"}
# Passive markers in Chinese
PASSIVE_MARKERS = {"被", "叫", "让", "给"}

# Helper function to get valid lemma or text if lemma is empty
def get_valid_lemma(token):
    return token.text if not token.lemma_ or token.lemma_ == '' else token.lemma_

# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet, depTagSet):
    # Chinese conjunctions
    lexical_match = ("和" in depSet or "与" in depSet or "及" in depSet or 
                    "或" in depSet or "或者" in depSet or "但" in depSet or 
                    "但是" in depSet or "而" in depSet or "并且" in depSet)
    dependency_match = ("conj" in depTagSet or "cc" in depTagSet)
    return lexical_match or dependency_match

def contains_conj_strict(depSet, depTagSet):
    # Chinese conjunctions - strict match
    lexical_match = ("和" in depSet or "与" in depSet or "及" in depSet or 
                    "或" in depSet or "或者" in depSet or "但" in depSet or 
                    "但是" in depSet or "而" in depSet or "并且" in depSet)
    return lexical_match


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        rightDepTags = {tok.dep_ for tok in rights}
        if contains_conj(rightDeps, rightDepTags):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ in NOUN_POS or tok.dep_ == "conj"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        rightDepTags = {tok.dep_ for tok in rights}
        if contains_conj(rightDeps, rightDepTags):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ in NOUN_POS or tok.dep_ == "conj"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies - adapted for Chinese
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and (head.pos_ not in NOUN_POS) and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ in SUBJECTS]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ in NOUN_POS:
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated? - adapted for Chinese negations
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_ in OBJECTS or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# xcomp; open complement - verb has no subject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    # In Chinese, subjects are typically to the left of the verb
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# find the main verb - or any aux verb if we can't find it
def _find_verbs(tokens):
    # For Chinese, we need to consider all verbs
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux" and tok.dep_ != "auxpass" and tok.dep_ != "aux:asp"]
    
    # If no verbs found, try more relaxed criteria
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    
    # If still no verbs found, try even more relaxed criteria
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    
    return verbs


# is the token an auxiliary verb?
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass" and tok.dep_ != "aux:asp")


# is the token a verb?  (including auxiliary verbs)
def _is_verb(tok):
    return tok.pos_ == "VERB" or tok.pos_ == "AUX"


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. 他打了并伤害了我)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


# return true if the current verb is passive - adapted for Chinese passive markers
def _is_cur_v_passive(v):
    # Check for Chinese passive markers like "被", "叫", "让", "给"
    for tok in v.children:
        if tok.dep_ == "auxpass" or tok.lower_ in PASSIVE_MARKERS:
            return True
    
    # Also check if any of the left children are passive markers
    for tok in v.lefts:
        if tok.lower_ in PASSIVE_MARKERS:
            return True
    
    return False


# Check if a list of indices is continuous
def check_continuous(index_list):
    if not index_list:
        return False
    
    sorted_indices = sorted(index_list)
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] != sorted_indices[i-1] + 1:
            return False
    
    return True


# Expand a token to include its children
def expand(item, tokens, visited, first=True):
    if item in visited:
        return []
    
    visited.add(item)
    results = [item]
    
    # Get all children of the item
    children = list(item.children)
    
    # Sort children by their position in the sentence
    children.sort(key=lambda x: x.i)
    
    # Add children to results if they are not breakers
    for child in children:
        if child.pos_ not in BREAKER_POS or (first and child.pos_ == "CCONJ"):
            child_results = expand(child, tokens, visited, False)
            results.extend(child_results)
    
    # Sort results by their position in the sentence
    results.sort(key=lambda x: x.i)
    
    # Check if the indices are continuous
    indices = [token.i for token in results]
    if not check_continuous(indices):
        # If not continuous, just return the original token
        return [item]
    
    return results


# Convert token list to string
def to_str(tokens):
    if not tokens:
        return None
    return ["".join([token.text for token in tokens]), [token.i for token in tokens]]


# Main function to find SVOs in Chinese text
def findSVOs(tokens):
    svos = []
    debug_info = []
    
    verbs = _find_verbs(tokens)
    # 移除调试输出
    # debug_info.append(f"找到的动词: {[(v.text, v.pos_, v.dep_) for v in verbs]}")
    
    for v in verbs:
        is_pas = _is_cur_v_passive(v)
        # 移除调试输出
        # debug_info.append(f"动词 '{v.text}' 是被动态: {is_pas}")
        
        subs, verbNegated = _get_all_subs(v)
        # 移除调试输出
        # debug_info.append(f"动词 '{v.text}' 的主语: {[s.text for s in subs]}")
        
        # hopefully there are subs
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                # 移除调试输出
                # debug_info.append(f"动词 '{v.text}' 有连接动词: {conjV.text}")
                v2, objs = _get_all_objs(conjV, is_pas)
                # 移除调试输出
                # debug_info.append(f"连接动词 '{conjV.text}' 的宾语: {[o.text for o in objs]}")
                
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        visited = set()
                        expanded_sub = expand(sub, tokens, visited)
                        expanded_obj = expand(obj, tokens, visited)
                        # 使用get_valid_lemma函数确保获取有效的词元
                        normalized_verb1 = "!" + get_valid_lemma(v) if verbNegated or objNegated else get_valid_lemma(v)
                        normalized_verb2 = "!" + get_valid_lemma(v2) if verbNegated or objNegated else get_valid_lemma(v2)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expanded_obj), [normalized_verb1, v.i], to_str(expanded_sub)))
                            svos.append((to_str(expanded_obj), [normalized_verb2, v2.i], to_str(expanded_sub)))
                        else:
                            svos.append((to_str(expanded_sub), [normalized_verb1, v.i],  to_str(expanded_obj)))
                            svos.append((to_str(expanded_sub), [normalized_verb2, v2.i],  to_str(expanded_obj)))
            else:
                v, objs = _get_all_objs(v, is_pas)
                # 移除调试输出
                # debug_info.append(f"动词 '{v.text}' 的宾语: {[o.text for o in objs]}")
                
                for sub in subs:
                    if len(objs) > 0:
                        for obj in objs:
                            objNegated = _is_negated(obj)
                            visited = set()
                            expanded_sub = expand(sub, tokens, visited)
                            expanded_obj = expand(obj, tokens, visited)
                            # 使用get_valid_lemma函数确保获取有效的词元
                            normalized_verb = "!" + get_valid_lemma(v) if verbNegated or objNegated else get_valid_lemma(v)
                            if is_pas:  # reverse object / subject for passive
                                svos.append((to_str(expanded_obj), [normalized_verb, v.i], to_str(expanded_sub)))
                            else:
                                svos.append((to_str(expanded_sub), [normalized_verb, v.i], to_str(expanded_obj)))
                    else:
                        # no obj - just return the SV parts
                        visited = set()
                        expanded_sub = expand(sub, tokens, visited)
                        # 使用get_valid_lemma函数确保获取有效的词元
                        normalized_verb = "!" + get_valid_lemma(v) if verbNegated else get_valid_lemma(v)
                        if is_pas:
                            svos.append((None, [normalized_verb, v.i], to_str(expanded_sub)))
                        else:
                            svos.append((to_str(expanded_sub), [normalized_verb, v.i], None))

        else:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                # 移除调试输出
                # debug_info.append(f"动词 '{v.text}' 有连接动词但没有主语: {conjV.text}")
                v2, objs = _get_all_objs(conjV, is_pas)
                # 移除调试输出
                # debug_info.append(f"连接动词 '{conjV.text}' 的宾语: {[o.text for o in objs]}")
                
                for obj in objs:
                    objNegated = _is_negated(obj)
                    visited = set()
                    expanded_obj = expand(obj, tokens, visited)
                    # 使用get_valid_lemma函数确保获取有效的词元
                    normalized_verb1 = "!" + get_valid_lemma(v) if verbNegated or objNegated else get_valid_lemma(v)
                    normalized_verb2 = "!" + get_valid_lemma(v2) if verbNegated or objNegated else get_valid_lemma(v2)
                    
                    if is_pas:  # reverse object / subject for passive
                        svos.append((to_str(expanded_obj), [normalized_verb1, v.i], None))
                        svos.append((to_str(expanded_obj), [normalized_verb2, v2.i], None))
                    else:
                        svos.append((None, [normalized_verb1, v.i], to_str(expanded_obj)))
                        svos.append((None, [normalized_verb2, v2.i], to_str(expanded_obj)))
            else:
                v, objs = _get_all_objs(v, is_pas)
                # 移除调试输出
                # debug_info.append(f"动词 '{v.text}' 没有主语但有宾语: {[o.text for o in objs]}")
                
                if len(objs) > 0:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        visited = set()
                        expanded_obj = expand(obj, tokens, visited)
                        # 使用get_valid_lemma函数确保获取有效的词元
                        normalized_verb = "!" + get_valid_lemma(v) if verbNegated or objNegated else get_valid_lemma(v)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expanded_obj), [normalized_verb, v.i], None))
                        else:
                            svos.append((None, [normalized_verb, v.i], to_str(expanded_obj)))

    # Filter out None values and duplicates
    filtered_svos = []
    seen = set()
    for svo in svos:
        # Skip if both subject and object are None
        if svo[0] is None and svo[2] is None:
            continue
            
        # Create a hashable representation of the SVO triplet
        hashable = (
            str(svo[0][0]) if svo[0] else "None", 
            svo[1][0], 
            str(svo[2][0]) if svo[2] else "None"
        )
        if hashable not in seen:
            seen.add(hashable)
            filtered_svos.append(svo)
    
    # 移除调试信息输出
    # for info in debug_info:
    #     print(info)
    
    return filtered_svos


if __name__ == '__main__':
    # Chinese test cases
    test_cases = [
        [0, 1, "中国科学家正在研究新型冠状病毒的特性和传播途径。"],
        [1, 2, "患者被感染了新型冠状病毒，目前正在医院接受治疗。"],
        [2, 3, "北京大学和清华大学联合发表了一篇关于人工智能的重要论文。"],
        [3, 4, "上海市政府宣布了一系列新的环保措施，以改善空气质量。"],
        [4, 5, "李明和王华一起开发了这个创新的软件系统。"],
        [5, 6, "这项技术被广泛应用于医疗、教育和工业领域。"]
    ]

    for test_case in test_cases:
        raw_sent = test_case[2]
        doc = nlp(raw_sent)
        # 移除测试输出
        # print("\n" + "="*80)
        # print(raw_sent)
        # print("词性标注:", [(tok.text, tok.pos_, tok.dep_) for tok in doc])
        # print("-"*40)
        svos = findSVOs(doc)
        
        # 移除三元组显示
        # print("三元组:")
        # for svo in svos:
        #     subj = svo[0][0] if svo[0] else "None"
        #     verb_idx = svo[1][1]  # Get the verb index
        #     verb_text = doc[verb_idx].text
        #     obj = svo[2][0] if svo[2] else "None"
        #     print(f"  [{subj}, {verb_text}, {obj}]")
        #     
        # print("="*80)
