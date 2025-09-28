package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf16"
	"unicode/utf8"
)

var DEFAULT_STOPWORDS = map[string]bool{
	"the": true, "and": true, "of": true, "to": true, "in": true, "a": true, "is": true,
	"for": true, "on": true, "with": true, "by": true, "at": true, "this": true,
	"that": true, "are": true, "be": true, "was": true, "an": true,
}

var TEXT_EXTENSIONS = map[string]bool{
	".txt": true, ".csv": true, ".json": true, ".xml": true, ".html": true, ".md": true,
	".py": true, ".go": true, ".log": true,
}

type FileAnalyzer struct {
	FilePath         string
	ShowProgress     bool
	Encoding         string
	TopWords         int
	ExcludeStopwords bool
	Stopwords        map[string]bool
	MaxMemoryUsage   int
	mu               sync.Mutex
	Results          FileResults
	WordCounts       map[string]int
	Error            string
	uniqueWordsCount int32
}

type FileResults struct {
	Lines       int
	Words       int
	Chars       int
	Bytes       int
	LongestLine int
	WordFreq    []WordFrequency
	Encoding    string
	Empty       bool
}

type WordFrequency struct {
	Word  string
	Count int
}

type AnalysisConfig struct {
	Files            []string
	Progress         bool
	Words            bool
	TopWords         int
	Threads          int
	Encoding         string
	Recursive        bool
	Output           string
	OutputFile       string
	SortBy           string
	ExcludeStopwords bool
	StopwordsFile    string
	StrictMime       bool
	MaxWords         int
	Individual       bool
}

func main() {
	startTime := time.Now()

	config := parseFlags()
	if err := validateConfig(config); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	validFiles := getFiles(config)
	if len(validFiles) == 0 {
		log.Fatal("No valid or readable text files found.")
	}

	stopwords := loadStopwords(config.StopwordsFile, config.ExcludeStopwords)
	fmt.Printf("Analyzing %d file(s) with %d thread(s)...\n", len(validFiles), config.Threads)

	analyzers := make([]*FileAnalyzer, len(validFiles))
	for i, file := range validFiles {
		analyzers[i] = NewFileAnalyzer(file, config, stopwords)
	}

	results := analyzeFiles(analyzers, config)

	for _, result := range results {
		if result.Error != "" {
			fmt.Printf("Error: %s\n", result.Error)
		}
	}

	if config.Individual {
		printIndividualResults(results, config.Words)
	}

	printSummary(results, config)

	if config.Output != "" {
		saveResults(results, config)
	}

	fmt.Printf("\nTime taken: %.2fs\n", time.Since(startTime).Seconds())
}

func NewFileAnalyzer(filePath string, config *AnalysisConfig, stopwords map[string]bool) *FileAnalyzer {
	encoding := config.Encoding
	if encoding == "" {
		encoding = detectEncoding(filePath)
	}

	return &FileAnalyzer{
		FilePath:         filePath,
		ShowProgress:     config.Progress,
		Encoding:         encoding,
		TopWords:         config.TopWords,
		ExcludeStopwords: config.ExcludeStopwords,
		Stopwords:        stopwords,
		MaxMemoryUsage:   config.MaxWords,
		WordCounts:       make(map[string]int),
		Results: FileResults{Encoding: encoding},
	}
}

func (fa *FileAnalyzer) Analyze() error {
	file, err := os.Open(fa.FilePath)
	if err != nil {
		return fmt.Errorf("IOError: %v", err)
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return err
	}
	if info.Size() == 0 {
		fa.Results.Empty = true
		return nil
	}

	scanner := bufio.NewScanner(file)
	lineCount := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		fa.processLine(line)
		lineCount++
		if fa.ShowProgress && lineCount%1000 == 0 {
			fmt.Printf("\rProcessing %s: %d lines", filepath.Base(fa.FilePath), lineCount)
		}
	}
	if fa.ShowProgress {
		fmt.Printf("\rProcessing %s: done                \n", filepath.Base(fa.FilePath))
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	fa.finalizeWordFreq()
	return nil
}

func (fa *FileAnalyzer) processLine(line []byte) {
	fa.mu.Lock()
	defer fa.mu.Unlock()

	fa.Results.Bytes += len(line)

	decodedLine, err := decodeLine(line, fa.Encoding)
	if err != nil {
		if fa.Error == "" {
			fa.Error = fmt.Sprintf("Encoding error: %v", err)
		}
		return
	}

	fa.Results.Chars += utf8.RuneCountInString(decodedLine)
	fa.Results.Lines++

	stripped := strings.TrimRight(decodedLine, "\r\n")
	if len(stripped) > fa.Results.LongestLine {
		fa.Results.LongestLine = len(stripped)
	}

	words := extractWords(decodedLine)
	fa.Results.Words += len(words)

	if fa.ExcludeStopwords {
		words = filterStopwords(words, fa.Stopwords)
	}

	if int(atomic.LoadInt32(&fa.uniqueWordsCount)) < fa.MaxMemoryUsage {
		for _, word := range words {
			if atomic.LoadInt32(&fa.uniqueWordsCount) >= int32(fa.MaxMemoryUsage) {
				break
			}
			if _, exists := fa.WordCounts[word]; !exists {
				atomic.AddInt32(&fa.uniqueWordsCount, 1)
			}
			fa.WordCounts[word]++
		}
	}
}

func (fa *FileAnalyzer) finalizeWordFreq() {
	fa.mu.Lock()
	defer fa.mu.Unlock()

	if len(fa.WordCounts) == 0 {
		return
	}

	type wordCount struct {
		word  string
		count int
	}
	counts := make([]wordCount, 0, len(fa.WordCounts))
	for word, count := range fa.WordCounts {
		counts = append(counts, wordCount{word, count})
	}

	sort.Slice(counts, func(i, j int) bool {
		return counts[i].count > counts[j].count ||
			(counts[i].count == counts[j].count && counts[i].word < counts[j].word)
	})

	maxWords := min(fa.TopWords, len(counts))
	fa.Results.WordFreq = make([]WordFrequency, maxWords)
	for i := 0; i < maxWords; i++ {
		fa.Results.WordFreq[i] = WordFrequency{Word: counts[i].word, Count: counts[i].count}
	}

	fa.WordCounts = nil
}

func analyzeFiles(analyzers []*FileAnalyzer, config *AnalysisConfig) []*FileAnalyzer {
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, config.Threads)
	results := make([]*FileAnalyzer, len(analyzers))

	for i, analyzer := range analyzers {
		wg.Add(1)
		semaphore <- struct{}{}
		go func(idx int, fa *FileAnalyzer) {
			defer wg.Done()
			defer func() { <-semaphore }()
			if err := fa.Analyze(); err != nil {
				fa.Error = err.Error()
			}
			results[idx] = fa
		}(i, analyzer)
	}

	wg.Wait()
	return results
}

func detectEncoding(filePath string) string {
	file, err := os.Open(filePath)
	if err != nil {
		return "utf-8"
	}
	defer file.Close()

	buffer := make([]byte, 1024)
	n, err := file.Read(buffer)
	if err != nil || n == 0 {
		return "utf-8"
	}
	buffer = buffer[:n]

	if n >= 2 && buffer[0] == 0xFE && buffer[1] == 0xFF {
		return "utf-16be"
	}
	if n >= 2 && buffer[0] == 0xFF && buffer[1] == 0xFE {
		return "utf-16le"
	}
	if isUTF8(buffer) {
		return "utf-8"
	}
	return "iso-8859-1"
}

func isUTF8(buffer []byte) bool {
	for i := 0; i < len(buffer); {
		if buffer[i] < 0x80 {
			i++
			continue
		}
		if buffer[i] < 0xC2 || buffer[i] > 0xF4 {
			return false
		}
		n := 0
		switch {
		case buffer[i] < 0xE0:
			n = 1
		case buffer[i] < 0xF0:
			n = 2
		default:
			n = 3
		}
		if i+n >= len(buffer) {
			return false
		}
		for j := 0; j < n; j++ {
			if buffer[i+1+j]&0xC0 != 0x80 {
				return false
			}
		}
		i += n + 1
	}
	return true
}

func decodeLine(line []byte, encoding string) (string, error) {
	switch encoding {
	case "utf-8":
		return string(line), nil
	case "utf-16le":
		if len(line)%2 != 0 {
			return "", fmt.Errorf("invalid UTF-16LE length")
		}
		u16 := make([]uint16, len(line)/2)
		for i := 0; i < len(line); i += 2 {
			u16[i/2] = uint16(line[i]) | uint16(line[i+1])<<8
		}
		return string(utf16.Decode(u16)), nil
	case "utf-16be":
		if len(line)%2 != 0 {
			return "", fmt.Errorf("invalid UTF-16BE length")
		}
		u16 := make([]uint16, len(line)/2)
		for i := 0; i < len(line); i += 2 {
			u16[i/2] = uint16(line[i+1]) | uint16(line[i])<<8
		}
		return string(utf16.Decode(u16)), nil
	default:
		var result strings.Builder
		for _, b := range line {
			result.WriteRune(rune(b))
		}
		return result.String(), nil
	}
}

var wordRegex = regexp.MustCompile(`[\p{L}\p{N}]+(?:['-][\p{L}\p{N}]+)*`)

func extractWords(text string) []string {
	return wordRegex.FindAllString(strings.ToLower(text), -1)
}

func filterStopwords(words []string, stopwords map[string]bool) []string {
	if stopwords == nil {
		return words
	}
	filtered := make([]string, 0, len(words))
	for _, word := range words {
		if !stopwords[word] {
			filtered = append(filtered, word)
		}
	}
	return filtered
}

func loadStopwords(stopwordsFile string, excludeStopwords bool) map[string]bool {
	if !excludeStopwords {
		return nil
	}
	if stopwordsFile == "" {
		return DEFAULT_STOPWORDS
	}

	file, err := os.Open(stopwordsFile)
	if err != nil {
		fmt.Printf("Warning: Could not load stopwords from %s, using defaults: %v\n", stopwordsFile, err)
		return DEFAULT_STOPWORDS
	}
	defer file.Close()

	stopwords := make(map[string]bool)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		word := strings.TrimSpace(strings.ToLower(scanner.Text()))
		if word != "" {
			stopwords[word] = true
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Printf("Warning: Error reading stopwords file, using defaults: %v\n", err)
		return DEFAULT_STOPWORDS
	}
	return stopwords
}

func isTextFile(filePath string, strictMime bool) bool {
	file, err := os.Open(filePath)
	if err != nil {
		return false
	}
	defer file.Close()

	if !strictMime {
		ext := strings.ToLower(filepath.Ext(filePath))
		if TEXT_EXTENSIONS[ext] {
			return true
		}
	}

	info, err := file.Stat()
	if err != nil || info.Size() == 0 {
		return false
	}

	buffer := make([]byte, min(1024, int(info.Size())))
	_, err = file.Read(buffer)
	if err != nil {
		return false
	}

	printable := 0
	for _, b := range buffer {
		if (b >= 32 && b <= 126) || b == 9 || b == 10 || b == 13 {
			printable++
		}
	}
	return float64(printable)/float64(len(buffer)) > 0.7
}

func getFiles(config *AnalysisConfig) []string {
	validFiles := make(map[string]bool)
	for _, pattern := range config.Files {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			fmt.Printf("Error processing pattern '%s': %v\n", pattern, err)
			continue
		}
		for _, filePath := range matches {
			if config.Recursive {
				filepath.Walk(filePath, func(path string, info os.FileInfo, err error) error {
					if err != nil || info.IsDir() {
						return nil
					}
					if isTextFile(path, config.StrictMime) {
						validFiles[path] = true
					}
					return nil
				})
			} else {
				if isTextFile(filePath, config.StrictMime) {
					validFiles[filePath] = true
				}
			}
		}
	}

	files := make([]string, 0, len(validFiles))
	for file := range validFiles {
		files = append(files, file)
	}
	sort.Strings(files)
	return files
}

func printIndividualResults(analyzers []*FileAnalyzer, showWordFreq bool) {
	for _, analyzer := range analyzers {
		fmt.Printf("\nFile: %s\n", analyzer.FilePath)
		if analyzer.Error != "" {
			fmt.Printf("  Error: %s\n", analyzer.Error)
			continue
		}
		fmt.Printf("  Lines: %d\n", analyzer.Results.Lines)
		fmt.Printf("  Words: %d\n", analyzer.Results.Words)
		fmt.Printf("  Chars: %d\n", analyzer.Results.Chars)
		fmt.Printf("  Bytes: %d\n", analyzer.Results.Bytes)
		fmt.Printf("  Longest Line: %d chars\n", analyzer.Results.LongestLine)
		fmt.Printf("  Encoding: %s\n", analyzer.Results.Encoding)
		if showWordFreq && len(analyzer.Results.WordFreq) > 0 {
			fmt.Printf("\n  --- Top %d Words ---\n", len(analyzer.Results.WordFreq))
			for _, wf := range analyzer.Results.WordFreq {
				fmt.Printf("  %s: %d\n", wf.Word, wf.Count)
			}
		}
	}
}

func printSummary(analyzers []*FileAnalyzer, config *AnalysisConfig) {
	successful := make([]*FileAnalyzer, 0, len(analyzers))
	for _, analyzer := range analyzers {
		if analyzer.Error == "" {
			successful = append(successful, analyzer)
		}
	}
	if len(successful) == 0 {
		fmt.Println("No files were successfully analyzed.")
		return
	}

	fmt.Println("\n=== Summary ===")
	headers := []string{"File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Encoding"}
	if config.Words {
		headers = append(headers, "Top Words")
	}

	widths := make([]int, len(headers))
	for i, header := range headers {
		widths[i] = len(header)
	}
	table := make([][]string, len(successful))
	totals := [4]int{}

	for i, analyzer := range successful {
		row := []string{
			filepath.Base(analyzer.FilePath),
			strconv.Itoa(analyzer.Results.Lines),
			strconv.Itoa(analyzer.Results.Words),
			strconv.Itoa(analyzer.Results.Chars),
			strconv.Itoa(analyzer.Results.Bytes),
			fmt.Sprintf("%d chars", analyzer.Results.LongestLine),
			analyzer.Results.Encoding,
		}
		if config.Words {
			topWords := "N/A"
			if len(analyzer.Results.WordFreq) > 0 {
				var words []string
				for _, wf := range analyzer.Results.WordFreq[:min(3, len(analyzer.Results.WordFreq))] {
					words = append(words, fmt.Sprintf("%s:%d", wf.Word, wf.Count))
				}
				topWords = strings.Join(words, ", ")
				if len(analyzer.Results.WordFreq) > 3 {
					topWords += "..."
				}
			}
			row = append(row, topWords)
		}
		table[i] = row
		for j, cell := range row {
			widths[j] = max(widths[j], len(cell))
		}
		totals[0] += analyzer.Results.Lines
		totals[1] += analyzer.Results.Words
		totals[2] += analyzer.Results.Chars
		totals[3] += analyzer.Results.Bytes
	}

	if config.SortBy != "" {
		sortIndex := map[string]int{
			"lines": 1, "words": 2, "chars": 3, "bytes": 4, "longest_line": 5,
		}[config.SortBy]
		sort.Slice(table, func(i, j int) bool {
			valI := parseSortValue(table[i][sortIndex])
			valJ := parseSortValue(table[j][sortIndex])
			return valI > valJ
		})
	}

	printRow(headers, widths)
	fmt.Println(strings.Repeat("-", sum(widths)+3*(len(headers)-1)))
	for _, row := range table {
		printRow(row, widths)
	}
	printRow([]string{"TOTAL", strconv.Itoa(totals[0]), strconv.Itoa(totals[1]), strconv.Itoa(totals[2]), strconv.Itoa(totals[3]), "", ""}, widths)
}

func parseSortValue(s string) int {
	s = strings.TrimSuffix(s, " chars")
	val, _ := strconv.Atoi(s)
	return val
}

func saveResults(analyzers []*FileAnalyzer, config *AnalysisConfig) {
	file, err := os.Create(config.OutputFile)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer file.Close()

	switch config.Output {
	case "csv":
		writer := csv.NewWriter(file)
		defer writer.Flush()
		headers := []string{"File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Encoding"}
		if config.Words {
			headers = append(headers, "Top Words")
		}
		writer.Write(headers)
		for _, analyzer := range analyzers {
			if analyzer.Error != "" {
				continue
			}
			row := []string{
				filepath.Base(analyzer.FilePath),
				strconv.Itoa(analyzer.Results.Lines),
				strconv.Itoa(analyzer.Results.Words),
				strconv.Itoa(analyzer.Results.Chars),
				strconv.Itoa(analyzer.Results.Bytes),
				fmt.Sprintf("%d", analyzer.Results.LongestLine),
				analyzer.Results.Encoding,
			}
			if config.Words && len(analyzer.Results.WordFreq) > 0 {
				var words []string
				for _, wf := range analyzer.Results.WordFreq {
					words = append(words, fmt.Sprintf("%s:%d", wf.Word, wf.Count))
				}
				row = append(row, strings.Join(words, ", "))
			}
			writer.Write(row)
		}
	case "json":
		enc := json.NewEncoder(file)
		enc.SetIndent("", "  ")
		if err := enc.Encode(analyzers); err != nil {
			fmt.Printf("Error encoding JSON: %v\n", err)
		}
	default:
		fmt.Println("Unsupported output format. Use 'csv' or 'json'.")
	}
}

func printRow(row []string, widths []int) {
	for i, cell := range row {
		fmt.Printf("%-*s", widths[i], cell)
		if i < len(row)-1 {
			fmt.Print(" | ")
		}
	}
	fmt.Println()
}

func parseFlags() *AnalysisConfig {
	config := &AnalysisConfig{}
	flag.BoolVar(&config.Progress, "progress", false, "Show progress during analysis")
	flag.BoolVar(&config.Words, "words", false, "Enable word frequency analysis")
	flag.IntVar(&config.TopWords, "top", 10, "Number of top words to show")
	flag.IntVar(&config.Threads, "threads", 1, "Number of threads to use")
	flag.StringVar(&config.Encoding, "encoding", "", "File encoding (default: auto-detect)")
	flag.BoolVar(&config.Recursive, "recursive", false, "Recursively analyze files in directories")
	flag.StringVar(&config.Output, "output", "", "Output format (csv or json)")
	flag.StringVar(&config.OutputFile, "out", "analysis_output", "Output file name")
	flag.StringVar(&config.SortBy, "sort", "", "Sort summary by (lines, words, chars, bytes, longest_line)")
	flag.BoolVar(&config.ExcludeStopwords, "exclude-stopwords", false, "Exclude common stopwords")
	flag.StringVar(&config.StopwordsFile, "stopwords-file", "", "Custom stopwords file")
	flag.BoolVar(&config.StrictMime, "strict-mime", false, "Strict MIME type checking")
	flag.IntVar(&config.MaxWords, "max-words", 1000000, "Maximum unique words in memory")
	flag.BoolVar(&config.Individual, "individual", false, "Show individual file analysis")
	flag.Parse()

	config.Files = flag.Args()
	return config
}

func validateConfig(config *AnalysisConfig) error {
	if len(config.Files) == 0 {
		return fmt.Errorf("no input files specified")
	}
	if config.TopWords <= 0 {
		return fmt.Errorf("top words must be greater than 0")
	}
	if config.Threads <= 0 {
		return fmt.Errorf("threads must be greater than 0")
	}
	if config.Output != "" && config.Output != "csv" && config.Output != "json" {
		return fmt.Errorf("invalid output format: %s", config.Output)
	}
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sum(nums []int) int {
	total := 0
	for _, n := range nums {
		total += n
	}
	return total
}
