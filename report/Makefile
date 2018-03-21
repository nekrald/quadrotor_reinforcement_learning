all:
	pdflatex description.tex
	bibtex description.aux
	pdflatex description.tex
	pdflatex description.tex
	okular description.pdf &

clean:
	rm -rf *.log *.aux *.bbl *.pdf *.blg
