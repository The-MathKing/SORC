import os
import re

def parse_results(filename):
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            dataset = parts[0]
            svd = parts[1]
            gcn = parts[2]
            afrc = parts[3]
            sorc = parts[4]
            results[dataset] = {"SVD": svd, "GCN": gcn, "AFRC": afrc, "SORC": sorc}
    return results

def patch_tex(results):
    with open('sorc_manuscript.tex', 'r') as f:
        tex = f.read()
        
    for ds in ["Amazon", "MovieLens"]:
        if ds in results:
            if ds == "Amazon":
                # Find line with Amazon Video Games & - & - & - & - \\
                pattern = r"(Amazon Video Games\s*&)\s*[-.0-9]+\s*&\s*[-.0-9]+\s*&\s*[-.0-9]+\s*&\s*[-.0-9]+\s*\\\\"
                replacement = f"\\1 {results[ds]['SVD']} & {results[ds]['GCN']} & {results[ds]['AFRC']} & \\textbf{{{results[ds]['SORC']}}} \\\\"
                tex = re.sub(pattern, replacement, tex)
            elif ds == "MovieLens":
                pattern = r"(MovieLens-1M\s*&)\s*[-.0-9]+\s*&\s*[-.0-9]+\s*&\s*[-.0-9]+\s*&\s*[-.0-9]+\s*\\\\"
                replacement = f"\\1 {results[ds]['SVD']} & {results[ds]['GCN']} & {results[ds]['AFRC']} & \\textbf{{{results[ds]['SORC']}}} \\\\"
                tex = re.sub(pattern, replacement, tex)
                
    with open('sorc_manuscript.tex', 'w') as f:
        f.write(tex)
        
if __name__ == "__main__":
    if os.path.exists('link_prediction_results.txt'):
        res = parse_results('link_prediction_results.txt')
        patch_tex(res)
        print("Patched sorc_manuscript.tex with results.")
    else:
        print("link_prediction_results.txt not found.")
