import os
import re

def parse_results(filename):
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 5: continue
            dataset = parts[0]
            # parse the mean value to determine which one is highest
            vals = []
            for p in parts[1:]:
                # extract the mean from "mean\pmstd" or "mean"
                mean_str = p.split('\\pm')[0]
                try:
                    vals.append(float(mean_str))
                except ValueError:
                    vals.append(0.0)
            
            best_idx = vals.index(max(vals))
            formatted = []
            for i, p in enumerate(parts[1:]):
                # Ensure it's treated as math mode to render \pm correctly
                math_p = f"${p}$"
                if i == best_idx:
                    formatted.append(f"\\textbf{{{math_p}}}")
                else:
                    formatted.append(math_p)
                    
            results[dataset] = {"SVD": formatted[0], "GCN": formatted[1], "AFRC": formatted[2], "SORC": formatted[3]}
    return results

def patch_tex(results):
    with open('sorc_manuscript.tex', 'r') as f:
        tex = f.read()
        
    for ds in ["Amazon", "MovieLens"]:
        if ds in results:
            if ds == "Amazon":
                pattern = r"(Amazon Video Games\s*&).*?\\\\"
                tex = re.sub(pattern, lambda m: f"{m.group(1)} {results[ds]['SVD']} & {results[ds]['GCN']} & {results[ds]['AFRC']} & {results[ds]['SORC']} \\\\", tex, flags=re.DOTALL)
            elif ds == "MovieLens":
                pattern = r"(MovieLens-1M\s*&).*?\\\\"
                tex = re.sub(pattern, lambda m: f"{m.group(1)} {results[ds]['SVD']} & {results[ds]['GCN']} & {results[ds]['AFRC']} & {results[ds]['SORC']} \\\\", tex, flags=re.DOTALL)
                
    with open('sorc_manuscript.tex', 'w') as f:
        f.write(tex)
        
if __name__ == "__main__":
    if os.path.exists('link_prediction_results.txt'):
        res = parse_results('link_prediction_results.txt')
        patch_tex(res)
        print("Patched sorc_manuscript.tex with results.")
    else:
        print("link_prediction_results.txt not found.")
