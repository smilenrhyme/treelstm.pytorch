import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.StringUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

public class MinimalConstituencyParse {

    private boolean tokenize;
    private BufferedWriter tokWriter, parentWriter;
    private LexicalizedParser parser;
    private TreeBinarizer binarizer;
    private CollapseUnaryTransformer transformer;

    private static final String PCFG_PATH = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";

    public MinimalConstituencyParse(String tokPath, String parentPath, boolean tokenize) throws IOException {
        this.tokenize = tokenize;
        if (tokPath != null) {
            tokWriter = new BufferedWriter(new FileWriter(tokPath));
        }
        parentWriter = new BufferedWriter(new FileWriter(parentPath));
        parser = LexicalizedParser.loadModel(PCFG_PATH);
        binarizer = TreeBinarizer.simpleTreeBinarizer(
                parser.getTLPParams().headFinder(), parser.treebankLanguagePack());
        transformer = new CollapseUnaryTransformer();
    }

    public List<HasWord> sentenceToTokens(String line) {
        List<HasWord> tokens = new ArrayList<>();
        if (tokenize) {
            PTBTokenizer<Word> tokenizer = new PTBTokenizer(new StringReader(line), new WordTokenFactory(), "");
            for (Word label; tokenizer.hasNext(); ) {
                tokens.add(tokenizer.next());
            }
        } else {
            for (String word : line.split(" ")) {
                tokens.add(new Word(word));
            }
        }

        return tokens;
    }

    public Tree parse(List<HasWord> tokens) {
        Tree tree = parser.apply(tokens);
        return tree;
    }

    public int[] constTreeParents(Tree tree) {
        Tree binarized = binarizer.transformTree(tree);
        Tree collapsedUnary = transformer.transformTree(binarized);
        Trees.convertToCoreLabels(collapsedUnary);
        collapsedUnary.indexSpans();
        List<Tree> leaves = collapsedUnary.getLeaves();
        int size = collapsedUnary.size() - leaves.size();
        int[] parents = new int[size];
        HashMap<Integer, Integer> index = new HashMap<Integer, Integer>();

        int idx = leaves.size();
        int leafIdx = 0;
        for (Tree leaf : leaves) {
            Tree cur = leaf.parent(collapsedUnary); // go to preterminal
            int curIdx = leafIdx++;
            boolean done = false;
            while (!done) {
                Tree parent = cur.parent(collapsedUnary);
                if (parent == null) {
                    parents[curIdx] = 0;
                    break;
                }

                int parentIdx;
                int parentNumber = parent.nodeNumber(collapsedUnary);
                if (!index.containsKey(parentNumber)) {
                    parentIdx = idx++;
                    index.put(parentNumber, parentIdx);
                } else {
                    parentIdx = index.get(parentNumber);
                    done = true;
                }

                parents[curIdx] = parentIdx + 1;
                cur = parent;
                curIdx = parentIdx;
            }
        }

        return parents;
    }

    public void printTokens(List<HasWord> tokens) throws IOException {
        int len = tokens.size();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len - 1; i++) {
            if (tokenize) {
                sb.append(PTBTokenizer.ptbToken2Text(tokens.get(i).word()));
            } else {
                sb.append(tokens.get(i).word());
            }
            sb.append(' ');
        }

        if (tokenize) {
            sb.append(PTBTokenizer.ptbToken2Text(tokens.get(len - 1).word()));
        } else {
            sb.append(tokens.get(len - 1).word());
        }

        sb.append('\n');
        tokWriter.write(sb.toString());
    }

    public void printParents(int[] parents) throws IOException {
        StringBuilder sb = new StringBuilder();
        int size = parents.length;
        for (int i = 0; i < size - 1; i++) {
            sb.append(parents[i]);
            sb.append(' ');
        }
        sb.append(parents[size - 1]);
        sb.append('\n');
        parentWriter.write(sb.toString());
    }

    public void close() throws IOException {
        if (tokWriter != null) tokWriter.close();
        parentWriter.close();
    }

    public static void main(String[] args) throws Exception {
        Properties props = StringUtils.argsToProperties(args);
        if (!props.containsKey("parentpath")) {
            System.err.println(
                    "usage: java MinimalConstituencyParse -tokenize - -tokpath <tokpath> -parentpath <parentpath> -sentence <sentence>");
            System.exit(1);
        }

        // whether to tokenize input sentences
        boolean tokenize = false;
        if (props.containsKey("tokenize")) {
            tokenize = true;
        }

        String tokPath = props.containsKey("tokpath") ? props.getProperty("tokpath") : null;
        String parentPath = props.getProperty("parentpath");
        MinimalConstituencyParse processor = new MinimalConstituencyParse(tokPath, parentPath, tokenize);

        List<HasWord> tokens = processor.sentenceToTokens(props.getProperty("sentence"));
        Tree parse = processor.parse(tokens);

        // produce parent pointer representation
        int[] parents = processor.constTreeParents(parse);

        // print
        if (tokPath != null) {
            processor.printTokens(tokens);
        }
        processor.printParents(parents);
        processor.close();
    }
}
