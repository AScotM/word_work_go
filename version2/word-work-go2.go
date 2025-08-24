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
	"unicode/utf8"
)

// Default stopwords for filtering common words.
var DEFAULT_STOPWORDS = map[string]bool{
	"the": true, "and": true, "of": true, "to": true, "in": true, "a": true, "is": true,
	"for": true, "on": true, "with": true, "by": true, "at": true, "this": true,
	"that": true, "are": true, "be": true, "was": true, "an": true,
}

// Supported text file extensions.
var TEXT_EXTENSIONS = map[string]bool{
	".txt": true, ".csv": true, ".json": true, ".xml": true, ".html": true, ".md": true,
	".py": true, ".go": true, ".log": true,
}

// FileAnalyzer holds data and configuration for analyzing a single file.
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

// FileResults stores analysis results for a file.
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

// WordFrequency represents a word and its frequency.
type WordFrequency struct {
	Word  string
	Count int
}

// AnalysisConfig holds configuration for the analysis process.
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

// main orchestrates the file analysis process.
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

	var errors []string
	for _, result := range results {
		if result.Error != "" {
			errors = append(errors, result.Error)
		}
	}

	for _, err := range errors {
		fmt.Printf("Error: %s\n", err)
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

// NewFileAnalyzer creates a new FileAnalyzer instance.
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
		Results: FileResults{
			Encoding: encoding,
		},
	}
}

// Analyze processes a file and populates analysis results.
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

// processLine analyzes a single line of text.
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

// finalizeWordFreq sorts and limits word frequencies.
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
		return counts[i].count > counts[j].count || (counts[i].count == counts[j].count && counts[i].word < counts[j].word)
	})

	maxWords := min(fa.TopWords, len(counts))
	fa.Results.WordFreq = make([]WordFrequency, maxWords)
	for i := 0; i < maxWords; i++ {
		fa.Results.WordFreq[i] = WordFrequency{
			Word:  counts[i].word,
			Count: counts[i].count,
		}
	}

	fa.WordCounts = nil // Free memory
}

// analyzeFiles processes files concurrently.
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
			results[idx] = fa // Safe due to pre-allocated slice
		}(i, analyzer)
	}

	wg.Wait()
	return results
}

// detectEncoding infers file encoding based on byte patterns.
func detectEncoding(filePath string) string {
	file, err := os.Open(filePath)
	if err != nil {
		return "utf-8"
	}
	defer file.Close()

	buffer := make([]byte, 1024)
	n, err := file.Read(buffer)
	if err != nil {
		return "utf-8"
	}
	buffer = buffer[:n]

	// Check for BOM or patterns
	if n >= 2 && buffer[0] == 0xFE && buffer[1] == 0xFF {
		return "utf-16be"
	}
	if n >= 2 && buffer[0] == 0xFF && buffer[1] == 0xFE {
		return "utf-16le"
	}
	if isUTF8(buffer) {
		return "utf-8"
	}
	if isWindows1252(buffer) {
		return "windows-1252"
	}
	return "iso-8859-1"
}

// isUTF8 checks if a buffer is likely UTF-8.
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

// isWindows1252 checks if a buffer is likely Windows-1252.
func isWindows1252(buffer []byte) bool {
	for _, b := range buffer {
		if (b >= 0x80 && b <= 0x8F) || (b >= 0x91 && b <= 0x9F) || b == 0xA0 {
			return true // Windows-1252-specific characters
		}
	}
	return false
}

// decodeLine converts a byte slice to a string based on encoding.
func decodeLine(line []byte, encoding string) (string, error) {
	switch encoding {
	case "utf-8":
		return string(line), nil
	case "utf-16le":
		if len(line)%2 != 0 {
			return "", fmt.Errorf("invalid UTF-16LE byte length")
		}
		var result strings.Builder
		for i := 0; i < len(line); i += 2 {
			r := rune(line[i]) | rune(line[i+1])<<8
			result.WriteRune(r)
		}
		return result.String(), nil
	case "utf-16be":
		if len(line)%2 != 0 {
			return "", fmt.Errorf("invalid UTF-16BE byte length")
		}
		var result strings.Builder
		for i := 0; i < len(line); i += 2 {
			r := rune(line[i+1]) | rune(line[i])<<8
			result.WriteRune(r)
		}
		return result.String(), nil
	case "windows-1252", "iso-8859-1":
		var result strings.Builder
		for _, b := range line {
			result.WriteRune(rune(b))
		}
		return result.String(), nil
	default:
		return string(line), nil
	}
}

// wordRegex matches words, including those with numbers and hyphens/apostrophes.
var wordRegex = regexp.MustCompile(`[\p{L}\p{N}]+(?:['-][\p{L}\p{N}]+)*`)

// extractWords extracts words from text, converting to lowercase.
func extractWords(text string) []string {
	lower := strings.ToLower(text)
	matches := wordRegex.FindAllString(lower, -1)
	return matches
}

// filterStopwords removes stopwords from a word list.
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

// loadStopwords loads stopwords from a file or uses defaults.
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

// isTextFile determines if a file is likely a text file.
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

// getFiles collects valid text files based on configuration.
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

// printIndividualResults displays results for each file.
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

// printSummary displays a summary table of results.
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

	// Calculate max width for each column
	widths := make([]int, len(headers))
	for i, header := range headers {
		widths[i] = len(header)
	}
	table := make([][]string, len(successful))
	totals := [4]int{} // lines, words, chars, bytes

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
			"lines":       1,
			"words":       2,
			"chars":       3,
			"bytes":       4,
			"longest_line": 5,
		}[config.SortBy]
		sort.Slice(table, func(i, j int) bool {
			valI, _ := strconv.Atoi(strings.TrimSuffix(table[i][sortIndex], " chars"))
			valJ, _ := strconv.Atoi(strings.TrimSuffix(table[j][sortIndex], " chars"))
			return valI > valJ
		})
	}

	// Print table with dynamic widths
	for i, header := range headers {
		fmt.Printf("%-*s  ", widths[i], header)
	}
	fmt.Println()
	for i := range headers {
		fmt.Print(strings.Repeat("-", widths[i]) + "  ")
	}
	fmt.Println()
	for _, row := range table {
		for i, cell := range row {
			fmt.Printf("%-*s  ", widths[i], cell)
		}
		fmt.Println()
	}

	fmt.Printf("\nTotal: %d lines, %d words, %d chars, %d bytes\n",
		totals[0], totals[1], totals[2], totals[3])
}

// saveResults saves analysis results to a file in JSON or CSV format.
func saveResults(analyzers []*FileAnalyzer, config *AnalysisConfig) {
	outputPath := config.OutputFile + "." + config.Output
	file, err := os.Create(outputPath)
	if err != nil {
		fmt.Printf("Error saving to %s: %v\n", outputPath, err)
		return
	}
	defer file.Close()

	if config.Output == "json" {
		data := make([]map[string]interface{}, len(analyzers))
		for i, analyzer := range analyzers {
			fileData := map[string]interface{}{
				"file":        analyzer.FilePath,
				"error":       analyzer.Error,
				"encoding":    analyzer.Results.Encoding,
				"lines":       analyzer.Results.Lines,
				"words":       analyzer.Results.Words,
				"chars":       analyzer.Results.Chars,
				"bytes":       analyzer.Results.Bytes,
				"longestLine": analyzer.Results.LongestLine,
			}
			if len(analyzer.Results.WordFreq) > 0 {
				wordFreq := make(map[string]int)
				for _, wf := range analyzer.Results.WordFreq {
					wordFreq[wf.Word] = wf.Count
				}
				fileData["wordFreq"] = wordFreq
			}
			data[i] = fileData
		}
		encoder := json.NewEncoder(file)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(data); err != nil {
			fmt.Printf("Error encoding JSON: %v\n", err)
		}
	} else if config.Output == "csv" {
		writer := csv.NewWriter(file)
		defer writer.Flush()
		headers := []string{"File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Encoding", "Error"}
		if config.Words {
			headers = append(headers, "WordFrequencies")
		}
		if err := writer.Write(headers); err != nil {
			fmt.Printf("Error writing CSV header: %v\n", err)
			return
		}
		for _, analyzer := range analyzers {
			row := []string{
				analyzer.FilePath,
				strconv.Itoa(analyzer.Results.Lines),
				strconv.Itoa(analyzer.Results.Words),
				strconv.Itoa(analyzer.Results.Chars),
				strconv.Itoa(analyzer.Results.Bytes),
				strconv.Itoa(analyzer.Results.LongestLine),
				analyzer.Results.Encoding,
				analyzer.Error,
			}
			if config.Words {
				var freqs []string
				for _, wf := range analyzer.Results.WordFreq {
					freqs = append(freqs, fmt.Sprintf("%s:%d", wf.Word, wf.Count))
				}
				row = append(row, strings.Join(freqs, ";"))
			}
			if err := writer.Write(row); err != nil {
				fmt.Printf("Error writing CSV row: %v\n", err)
			}
		}
	}
	fmt.Printf("Results saved to %s\n", outputPath)
}

// parseFlags parses and returns command-line flags.
func parseFlags() *AnalysisConfig {
	config := &AnalysisConfig{}
	flag.BoolVar(&config.Progress, "progress", false, "Show progress during analysis")
	flag.BoolVar(&config.Words, "words", false, "Include top word frequencies")
	flag.IntVar(&config.TopWords, "top-words", 10, "Number of top words to display")
	flag.IntVar(&config.Threads, "threads", 4, "Number of threads to use")
	flag.StringVar(&config.Encoding, "encoding", "", "Force specific encoding (utf-8, utf-16le, utf-16be, windows-1252, iso-8859-1)")
	flag.BoolVar(&config.Recursive, "recursive", false, "Recursively search for files")
	flag.StringVar(&config.Output, "output", "", "Output format (json or csv)")
	flag.StringVar(&config.OutputFile, "output-file", "wc_analysis_results", "Output file base name")
	flag.StringVar(&config.SortBy, "sort-by", "", "Sort summary by (lines, words, chars, bytes, longest_line)")
	flag.BoolVar(&config.ExcludeStopwords, "exclude-stopwords", false, "Exclude common stopwords")
	flag.StringVar(&config.StopwordsFile, "stopwords-file", "", "Custom stopwords file")
	flag.BoolVar(&config.StrictMime, "strict-mime", false, "Use strict MIME type detection")
	flag.IntVar(&config.MaxWords, "max-words", 1000000, "Maximum unique words to track per file")
	flag.BoolVar(&config.Individual, "individual", false, "Show individual file results")
	flag.Parse()
	config.Files = flag.Args()
	if config.Threads <= 0 {
		config.Threads = 4
	}
	return config
}

// validateConfig checks flag values for validity.
func validateConfig(config *AnalysisConfig) error {
	if config.Output != "" && config.Output != "json" && config.Output != "csv" {
		return fmt.Errorf("output must be 'json' or 'csv'")
	}
	if config.SortBy != "" {
		validSort := map[string]bool{"lines": true, "words": true, "chars": true, "bytes": true, "longest_line": true}
		if !validSort[config.SortBy] {
			return fmt.Errorf("sort-by must be one of: lines, words, chars, bytes, longest_line")
		}
	}
	if len(config.Files) == 0 {
		return fmt.Errorf("no input files specified")
	}
	return nil
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
