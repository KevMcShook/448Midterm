"""Build a report from saved measurements; never generate example scores."""
import argparse
from datetime import date
from pathlib import Path
import json
import html
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--out',type=Path,default=Path('report'))
    parser.add_argument('--author',default='Kevin Shook')
    parser.add_argument('--cleaning-report',type=Path,default=Path('data/cleaned/cleaning_report.json'))
    args = parser.parse_args()
    def load(name):
        return json.loads((args.results/name).read_text())
    metrics = {r['experiment']:r for r in load('all_metrics.json')}
    splits,config,prep,style,ci,deltas,baseline = [load(n) for n in ['split_summary.json','config.json','preprocessing.json','style_analysis.json','uncertainty.json','rq2_differences.json','baseline.json']]
    cleaning = json.loads(args.cleaning_report.read_text())
    pct = lambda x: f'{x*100:.2f}%'
    interval = lambda bounds: f'{bounds[0]*100:.2f}-{bounds[1]*100:.2f}%'
    args.out.mkdir(parents=True,exist_ok=True)
    font_dir = Path('/usr/share/fonts/truetype/dejavu')
    if (font_dir/'DejaVuSans.ttf').exists():
        for name,file in [('Body','DejaVuSans.ttf'),('Body-Bold','DejaVuSans-Bold.ttf'),('Body-Oblique','DejaVuSans-Oblique.ttf')]:
            pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
        pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Oblique')
        normal,bold = 'Body','Body-Bold'
    else:
        normal,bold = 'Helvetica','Helvetica-Bold'
    blue = colors.HexColor('#174A63')
    gray = colors.HexColor('#53616A')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Text',fontName=normal,fontSize=9.5,leading=13.2,spaceAfter=8,textColor=colors.HexColor('#20333C')))
    styles.add(ParagraphStyle(name='TitleCustom',fontName=bold,fontSize=26,leading=30,textColor=blue,spaceAfter=12))
    styles.add(ParagraphStyle(name='Section',fontName=bold,fontSize=17,leading=21,textColor=blue,spaceAfter=13))
    styles.add(ParagraphStyle(name='Sub',fontName=bold,fontSize=11.5,leading=15,textColor=blue,spaceBefore=6,spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallCustom',parent=styles['Text'],fontSize=8,leading=11,textColor=gray,spaceAfter=6))
    styles.add(ParagraphStyle(name='Cell',parent=styles['Text'],fontSize=8.2,leading=11,spaceAfter=0))
    styles.add(ParagraphStyle(name='CellHeader',parent=styles['Cell'],fontName=bold,textColor=colors.white))
    story,markdown = [],[]
    width = A4[0]-108
    def paragraph(text,kind='Text'):
        story.append(Paragraph(html.escape(text),styles[kind]))
        markdown.append(text+'\n')
    def heading(text,level=1):
        story.append(Paragraph(html.escape(text),styles['Section' if level==1 else 'Sub']))
        markdown.append(('#'*(level+1))+' '+text+'\n')
    def table(headers,rows,widths):
        cells = [[Paragraph(html.escape(str(x)),styles['CellHeader']) for x in headers]]+[[Paragraph(html.escape(str(x)),styles['Cell']) for x in row] for row in rows]
        t = Table(cells,colWidths=[width*w for w in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),blue),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F4F6'),colors.white]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#B9C7CF'))]))
        story.extend([t,Spacer(1,10)])
        markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        markdown.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows)
        markdown.append('')
    def figure(name,caption,scale=1):
        path = args.results/'figures'/name
        image = Image(str(path))
        natural_width,natural_height = image.imageWidth,image.imageHeight
        image.drawWidth = width*scale
        image.drawHeight = natural_height/natural_width*image.drawWidth
        story.append(image)
        paragraph(caption,'SmallCustom')
        markdown.append(f'![{caption}](../{args.results.name}/figures/{name})\n')
    def next_page():
        story.append(PageBreak())

    # Page 1: the problem and measured answer.
    paragraph('CMPSC 448  |  MIDTERM PROJECT  |  OCTOBER 2026','SmallCustom')
    story.append(Paragraph('Who Wrote It?',styles['TitleCustom']))
    paragraph('Identifying Claude, Gemini, and Qwen from their responses','Sub')
    paragraph(f'{args.author}  |  Individual submission  |  {date.today().isoformat()}','SmallCustom')
    markdown.insert(0,'# Who Wrote It? Identifying LLMs from Their Responses\n')
    heading('Problem and principal findings',2)
    paragraph('Can a classifier identify the LLM family behind a response, and does seeing the user prompt help? This study evaluates a small convolutional neural network (CNN) and a recurrent network using LSTM on 9,000 balanced examples from three model configurations. The target is the generating family, not response quality or the winner of an Arena comparison. RQ1 and RQ2 are required; RQ4 is investigated as an additional analysis.')
    paragraph(f"On 1,351 held-out responses grouped to prevent exact prompt, response, and session overlap, response-only accuracy was {pct(metrics['cnn_output']['test']['accuracy'])} for the CNN and {pct(metrics['lstm_output']['test']['accuracy'])} for the LSTM. Prompt-only accuracy was near one-third. Adding prompts produced little measured benefit. Removing punctuation and newline tokens greatly reduced performance, suggesting substantial sensitivity to structural cues under this perturbation.")
    table(['Architecture','Text supplied','Accuracy','Macro-F1','Accuracy 95% interval'],[
        [arch.upper(),{'input':'Prompt only','output':'Response only','combined':'Prompt + response'}[mode],pct(metrics[f'{arch}_{mode}']['test']['accuracy']),f"{metrics[f'{arch}_{mode}']['test']['macro_f1']:.4f}",interval(ci[f'{arch}_{mode}']['accuracy_95_interval'])]
        for arch in ['cnn','lstm'] for mode in ['input','output','combined']], [.15,.23,.14,.14,.34])
    paragraph('Table 1. All six required comparisons use the same split and vocabulary. Intervals use 2,000 bootstrap samples of held-out split groups; they do not capture training-seed variability.','SmallCustom')
    figure('main_results.png','Figure 1. Response-based models substantially exceed chance. Error bars are grouped bootstrap 95% intervals.',.97)

    # Page 2: data and safeguards.
    next_page()
    heading('Dataset curation and evaluation split')
    paragraph('The source is Arena Human Preference 140K, containing text-only comparisons collected from April to July 2025 [1]. All seven downloaded shards contain 135,634 comparisons. Each side supplies a model name and conversation. Preference votes are discarded because they are not the target for authorship attribution. Public-source usage follows the source terms: prompts are CC-BY-4.0 and outputs are subject to their providers\' terms.')
    table(['Curation stage','Count'],[
        ['Original comparisons','135,634'],['English, first evaluation only','57,759'],['Candidate responses from requested prefixes','52,179'],['Removed: explicit model self-identification heuristic','366'],['Removed: non-text or malformed content','15'],['Removed: duplicate model/prompt/response','58'],['Retained cleaned responses','51,740'],['Balanced modeling subset','9,000']], [.76,.24])
    paragraph('One example is the first user message and immediately following assistant response on an eligible side. Only source language "en" and evaluation_order 1 are used, avoiding later context whose models may change. Model prefixes are claude-, gemini-, and qwen; the alias qwq-32b is not included. Unicode and line endings are normalized, NUL characters removed, and outer whitespace trimmed. Internal formatting is retained. Explicit self-identification filtering is heuristic and can remove quoted statements as well as genuine self-identification.')
    table(['Family','Exact configuration','Examples'],[[family.title(),model,'3,000'] for family,model in cleaning['balanced_model_versions'].items()],[.15,.69,.16])
    paragraph('A deterministic SHA-256 ordering with seed label 448 selects 3,000 responses from each configuration. Restricting each family to one configuration avoids mixing version effects in the main comparison, but the findings do not establish attribution across every model in a family. The larger cleaned collection contains 23,996 Claude, 17,269 Gemini, and 10,475 Qwen responses.')
    table(['Partition','Claude','Gemini','Qwen','Total'],[[part.title(),str(splits[part]['class_counts']['claude']),str(splits[part]['class_counts']['gemini']),str(splits[part]['class_counts']['qwen']),str(splits[part]['total'])] for part in ['train','validation','test']],[.28,.18,.18,.18,.18])
    paragraph('Twenty stratified group folds define a fixed approximately 70/15/15 split: 14 training folds, three validation folds, and three test folds. Connected records sharing a normalized prompt, normalized exact response, or evaluation session stay together. Assertions verify no overlap in any of these identifiers. Prompt normalization for grouping is case-insensitive with collapsed whitespace; response grouping retains case. Near-duplicate paraphrases are not detected. Raw-file SHA-256 checks confirm that the source shards were not changed.')

    # Page 3: implementation.
    next_page()
    heading('CNN, LSTM, and training procedure')
    paragraph('Both networks are implemented directly with PyTorch layers [2], trained from random initialization, and predict three class logits. They use only text. Model names, IDs, timestamps, labels, and category tags are excluded from inputs. Four implementation checks cover vocabulary isolation, preservation of the response budget, padding invariance, and learning a simple synthetic class signal.')
    table(['Component','CNN','LSTM'],[
        ['Embedding','20,000 x 64; padding ID 0','20,000 x 64; padding ID 0'],
        ['Text encoder','1D convolutions: widths 3, 4, 5; 64 filters each; ReLU','One forward LSTM layer; hidden size 64'],
        ['Aggregation','Masked max pooling; concatenate 192 features','Hidden state at the last real token'],
        ['Prediction','Dropout 0.3; linear 192 to 3','Dropout 0.3; linear 64 to 3'],
        ['Trainable parameters',f"{metrics['cnn_output']['parameter_count']:,}",f"{metrics['lstm_output']['parameter_count']:,}"]],[.24,.38,.38])
    paragraph('Tokenization lowercases word tokens, keeps individual punctuation tokens, and inserts a dedicated newline token. A shared vocabulary of at most 20,000 tokens is fitted only to training prompts and responses; tokens occurring fewer than twice become unknown. No test text affects vocabulary construction. Prompt-only models receive up to 128 tokens; response-only models receive up to 256. Combined input is up to 128 prompt tokens, a separator, then up to 256 response tokens, preserving the response budget. Padding is masked in CNN pooling; the forward LSTM reads the state before padding.')
    paragraph(f"Training uses AdamW (learning rate 0.001, weight decay 0.01), cross-entropy, batch size 64, and gradient norm clipping at 1.0. Maximum training is 12 epochs, with patience three on validation macro-F1 and minimum improvement 0.00001. The best validation checkpoint is restored before test evaluation. Seed 448 and deterministic PyTorch operations are used on CPU with four threads. These settings were fixed without a test-set hyperparameter search. Training and validation together for the six runs took about {sum(r['train_seconds'] for r in metrics.values())/60:.1f} minutes, excluding preparation and analysis.")
    table(['Model','Prompt best/run','Response best/run','Combined best/run'],[[arch.upper(),*[f"{metrics[f'{arch}_{m}']['best_epoch']} / {metrics[f'{arch}_{m}']['epochs_run']}" for m in ['input','output','combined']]] for arch in ['cnn','lstm']],[.22,.26,.26,.26])
    paragraph('The CNN tests local token patterns at three widths and pools their strongest activations anywhere in the retained text. The LSTM updates a hidden state sequentially and classifies from the last retained token. These are compact supervised classifiers, not pretrained language models. Their different aggregation mechanisms can respond differently to introductions, punctuation patterns, and word order. Training curves and checkpoint diagnostics are provided in the appendix.')

    # Page 4: interpretation and errors.
    next_page()
    heading('RQ1 and RQ2: interpretation and errors')
    paragraph(f"RQ1: Both models identify the generating configuration well above chance from responses. The response-only CNN achieves {pct(metrics['cnn_output']['test']['accuracy'])} accuracy and {metrics['cnn_output']['test']['macro_f1']:.4f} macro-F1; the LSTM reaches {pct(metrics['lstm_output']['test']['accuracy'])} and {metrics['lstm_output']['test']['macro_f1']:.4f}. Macro-F1 averages class-wise F1 equally. A training-majority baseline predicts Qwen (tied with Gemini in training) and obtains {pct(baseline['test_accuracy'])} accuracy and {baseline['test_macro_f1']:.4f} macro-F1; uniformly random guessing has expected accuracy 33.33%.")
    paragraph(f"RQ2: Prompt-only CNN and LSTM accuracies are {pct(metrics['cnn_input']['test']['accuracy'])} and {pct(metrics['lstm_input']['test']['accuracy'])}. Their uncertainty intervals include chance. Adding the prompt changes CNN accuracy by {deltas['cnn']['combined_minus_output_accuracy']*100:+.2f} percentage points, with a paired group-bootstrap interval of {deltas['cnn']['paired_group_bootstrap_95_interval'][0]*100:+.2f} to {deltas['cnn']['paired_group_bootstrap_95_interval'][1]*100:+.2f}. For the LSTM, the change is {deltas['lstm']['combined_minus_output_accuracy']*100:+.2f} points. This run provides little evidence that prompts materially improve attribution. It does not show that prompt bias is absent from every dataset.")
    figure('confusion_matrices.png','Figure 2. Held-out confusion matrices. Rows are true families and columns are predictions; cells show counts, with color normalized by row.',.96)
    paragraph('The output-only CNN makes 90 errors: 26 Claude, 27 Gemini, and 37 Qwen examples are misclassified. The output-only LSTM makes 206 errors, including 92 Qwen errors; its largest off-diagonal cell is Qwen predicted as Gemini (50). Similar formatting between those families is a possible explanation, examined by RQ4 rather than established by the confusion matrix alone. Prompt-only CNN predictions favor Gemini while providing essentially no useful class separation.')
    paragraph(f"The models primarily see response openings: {prep['truncation']['test']['response_fraction_truncated']*100:.1f}% of test responses exceed 256 tokens, while {prep['truncation']['test']['prompt_fraction_truncated']*100:.1f}% of prompts exceed 128. This limits claims about whole-response stylistic fingerprints. A longer context window and repeated seeds would be useful follow-up experiments.")

    # Page 5: RQ4.
    next_page()
    heading('RQ4: what distinguishes the responses?')
    paragraph('Descriptive statistics show clear differences in length and presentation in this sample. Statistics below use all 9,000 balanced examples and the full saved responses; they were not used to choose neural hyperparameters. Type-token diversity is measured with a simple word tokenizer and depends on text length, so it should not be interpreted independently of length.')
    table(['Family','Median words','Mean words','Bold formatting','Code fences'],[[f.title(),f"{style['description'][f]['median_response_words']:.0f}",f"{style['description'][f]['mean_response_words']:.1f}",pct(style['description'][f]['bold_fraction']),pct(style['description'][f]['code_fence_fraction'])] for f in ['claude','gemini','qwen']],[.19,.20,.19,.23,.19])
    figure('style_analysis.png','Figure 3. Length distributions omit boxplot outlier points for readability. The underlying data are not trimmed.',.95)
    paragraph('Two auxiliary logistic regressions use the same training and test partitions. The length-only model uses log word count. The style model uses ten features: log word and character counts, type-token ratio, and per-word rates of lines, bold markers, code fences, headings, list lines, exclamation marks, and question marks. Standardization is fitted on training data only; C=1 and max_iter=1,000 are fixed. These classifiers see full-response information, so their scores are diagnostic and not an equal-context neural comparison.')
    table(['Diagnostic','Test accuracy','Macro-F1'],[
        ['Length-only logistic regression',pct(style['length_only_classifier']['accuracy']),f"{style['length_only_classifier']['macro_f1']:.4f}"],
        ['Style-feature logistic regression',pct(style['style_only_classifier']['accuracy']),f"{style['style_only_classifier']['macro_f1']:.4f}"],
        ['CNN, formatting tokens removed',pct(style['formatting_ablation']['cnn']['accuracy']),f"{style['formatting_ablation']['cnn']['macro_f1']:.4f}"],
        ['LSTM, formatting tokens removed',pct(style['formatting_ablation']['lstm']['accuracy']),f"{style['formatting_ablation']['lstm']['macro_f1']:.4f}"]],[.60,.20,.20])
    paragraph('For the ablation, retain the original 256-token response window, remove punctuation and newline tokens, and evaluate the frozen response-only models. No new text is brought into the window, no model is retrained, and test results are not used to retune settings. Both models lose substantial accuracy. This demonstrates sensitivity to the perturbation, but removing tokens also changes sequence positions, length, and distribution. It does not prove that formatting alone caused the original accuracy. A stronger follow-up would retrain both models on consistently normalized text and compare performance across multiple seeds.')

    # Page 6: lessons, reproducibility, references.
    next_page()
    heading('Limitations, lessons, and reproducibility')
    heading('Limits of the evidence',2)
    paragraph('The results concern three specific 2025 model configurations in a public Arena sample. Sampling balances labels but not topics, dates, generation settings, or response lengths. Source language and category labels are inherited and may contain errors. Exact-match grouping prevents identified leakage but not semantic near-duplicates. Model-identity filtering is incomplete and may remove benign quotations. Public conversations have not undergone a comprehensive new privacy review; raw user text is not reproduced in the report. A new domain, newer model version, or different prompting style could change accuracy. RQ3 domain transfer was not evaluated.')
    paragraph('Only one training seed and one grouped partition were run. The reported bootstrap intervals describe test-group sampling uncertainty for these fitted models; they do not include architecture selection, training instability, or domain shift. Combined-input results and the formatting ablation should therefore be interpreted as limited comparisons, not universal claims.')
    heading('Lessons supported by this work',2)
    paragraph('First, data splitting is part of the scientific question: shared prompts and repeated responses must be grouped before evaluation. Second, strong attribution can coexist with chance-level prompt-only results, so a prompt-only control helps diagnose confounding. Third, superficial presentation can be highly predictive; high accuracy alone does not establish deep linguistic differences. Fourth, validation curves reveal prompt-only overfitting that training loss would hide. Finally, reproducibility includes file formats and environment details: gzip Parquet was required for the installed R Arrow build, and equality checks confirmed that re-encoding changed no values. These are observations from the project, not claims about undocumented personal experiences.')
    heading('Reproducing and reviewing the submission',2)
    paragraph('The repository includes cleaning and training scripts, R exploration, implementation tests, saved split assignments, predictions, metrics, figures, and this report. The README documents source retrieval and commands. Large raw data and local checkpoints are excluded from Git; regenerated data retain the required LLM_name, LLM_Input, and LLM_output fields. The distributed cleaned file has a recorded SHA-256 hash; an encoding note links it to the value-identical file used in training. The report is generated from saved metrics so numbers can be traced to experiment outputs.')
    paragraph('Software: Python 3.12, PyTorch 2.14.1+cpu, NumPy 2.3.5, pandas 2.2.3, PyArrow 26.0.0, scikit-learn 1.9.1, Matplotlib 3.11.2, and ReportLab 4.4.9. R exploration was verified with R 4.6.1, Arrow 25.0.1, and data.table. Hardware: Intel Core i9-13900H, CPU training. No paid LLM API calls were used.')
    paragraph('Assistance disclosure: Codex assisted with data cleaning, model implementation, experiment execution, verification, and report drafting. Kevin Shook is the individual author and should review and understand the submitted work, consistent with the course guidelines.')
    heading('References',2)
    for reference in [
        '[1] Arena. Arena Human Preference 140K dataset card. https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k (accessed October 10, 2026).',
        '[2] PyTorch documentation. torch.nn.Conv1d and torch.nn.LSTM. https://docs.pytorch.org/docs/stable/nn.html',
        '[3] scikit-learn documentation. StratifiedGroupKFold. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html',
        '[4] CMPSC 448. Who Wrote It? Identifying LLMs from Their Responses. Midterm project handout, modified September 17, 2026.'
    ]:
        paragraph(reference,'SmallCustom')
    next_page()
    heading('Appendix: training diagnostics')
    paragraph('Each curve shows one seeded training run. Training loss is recorded during optimization with dropout active; validation loss is evaluated with dropout disabled. The two losses therefore differ in both data and evaluation mode. Checkpoints are selected by validation macro-F1 rather than the lowest plotted validation loss, and the test set is not used for early stopping.')
    figure('training_curves.png','Figure 4. Training and validation cross-entropy over epochs. Dotted lines identify the selected checkpoints.',.99)
    table(['Experiment','Best epoch','Validation F1','Test F1'],[[name,f"{r['best_epoch']}",f"{r['validation']['macro_f1']:.4f}",f"{r['test']['macro_f1']:.4f}"] for name,r in metrics.items()],[.37,.18,.23,.22])
    paragraph('The prompt-only losses illustrate why training fit is insufficient evidence of useful attribution: training loss declines while validation loss worsens. Early stopping retains an earlier checkpoint, but the held-out results still remain near chance. The output and combined configurations learn more useful patterns. Some reach the 12-epoch ceiling, so the experiment does not establish that either architecture is optimally tuned.')
    paragraph('Source-level checks verified unique example IDs, balanced family counts, Parquet round-trip equality, unchanged raw-file hashes, and nine independently checked prompt/response/model pairings. Split checks found no exact prompt, response, or session overlap across partitions. Checkpoint reloads were exercised by the formatting analysis. These checks address implementation and data consistency; they do not establish validity on future models or unseen domains.')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#C7D3D9'))
        canvas.line(48,34,A4[0]-48,34)
        canvas.setFont(normal,8)
        canvas.setFillColor(gray)
        canvas.drawString(48,22,'CMPSC 448  |  '+args.author)
        canvas.drawRightString(A4[0]-48,22,str(doc.page))
    document = SimpleDocTemplate(str(args.out/'project_report.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=44,title='Who Wrote It? Identifying LLMs from Their Responses',author=args.author)
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    (args.out/'project_report.md').write_text('\n'.join(markdown)+'\n')
    print(json.dumps({'report':str(args.out/'project_report.pdf'),'author':args.author}))

if __name__=='__main__':
    main()
