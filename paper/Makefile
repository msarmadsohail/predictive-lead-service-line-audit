# Manuscript build and arXiv upload.
#
# arXiv compiles the source, so the tarball ships .tex, .bbl and figures, but no
# build artefacts and no main.pdf. Analysis scripts go in anc/.

TEX     := main.tex
DIST    := dist
SCRIPTS := ../scripts
ANC     := paths.py extract.py extract_t0.py nyc_core.py fetch_mappluto.py join_pluto.py \
           statewide_screen.py screen_extras.py t0_t1.py estimate.py baseline2.py \
           paper_numbers.py make_figures.py make_fig_screen.py

.PHONY: all clean dist check ascii abstract

all: main.pdf

main.pdf: $(TEX) refs.bib fig/*.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# overfull boxes and undefined refs block; underfull is advisory
check: main.pdf
	@! grep -q 'Overfull\|undefined' main.log || \
	  { echo "REFUSING: overfull box or undefined reference"; \
	    grep 'Overfull\|undefined' main.log; exit 1; }
	@echo "underfull (advisory): $$(grep -c Underfull main.log)"
	@echo "build clean"

# arXiv abstract field: 1920 chars, ASCII
abstract: abstract-arxiv.txt
	@python3 -c "import sys; s=open('abstract-arxiv.txt',encoding='utf-8').read().strip(); \
	n=len(s); bad=[c for c in s if ord(c)>127]; \
	print(f'metadata abstract: {n} chars (limit 1920), non-ASCII: {len(bad)}'); \
	sys.exit(1) if n>1920 or bad else None"

# arXiv metadata is ASCII; every character here has a TeX spelling
ascii:
	@python3 -c "import sys; bad=0; \
	[bad := bad or [print(f'{f}: U+{ord(c):04X}') for c in open(f,encoding='utf-8').read() if ord(c)>127] \
	for f in ['$(TEX)','refs.bib']]; \
	sys.exit(1) if bad else print('ASCII only: $(TEX) refs.bib')"

dist: check ascii abstract
	rm -rf $(DIST) && mkdir -p $(DIST)/fig $(DIST)/anc
	perl -pe 's/(^|[^\\])%.*/\1%/' < $(TEX) > $(DIST)/$(TEX)
	cp refs.bib main.bbl $(DIST)/
	# only figures the manuscript actually includes
	@for f in $$(grep -oP '\\includegraphics(\[[^]]*\])?\{\K[^}]+' $(TEX)); do \
	  cp "$$f" $(DIST)/"$$f" || exit 1; done
	@for f in $(ANC); do cp $(SCRIPTS)/$$f $(DIST)/anc/ || exit 1; done
	@python3 -c "import sys; \
	bad=[(f,ord(c)) for f in sys.argv[1:] for c in open(f,encoding='utf-8').read() if ord(c)>127]; \
	sys.exit(f'non-ASCII in shipped script: {bad[0]}') if bad else None" $(DIST)/anc/*.py
	# arXiv is case-sensitive; the working filesystem is not
	@cd $(DIST) && for f in $$(grep -oP '\\includegraphics(\[[^]]*\])?\{\K[^}]+' $(TEX)); do \
	  test -f "$$f" || test -f "$$f.pdf" || { echo "MISSING (case?): $$f"; exit 1; }; done
	@cd $(DIST) && for f in $$(find . -type f); do \
	  echo "$$f" | grep -qP '^[./a-zA-Z0-9_+,=-]+$$' || { echo "BAD FILENAME: $$f"; exit 1; }; done
	@! grep -rnE 'TODO|FIXME|\[to be|XXX' $(DIST)/$(TEX) || \
	  { echo "REFUSING: placeholder in upload"; exit 1; }
	cd $(DIST) && tar czf ../arxiv-upload.tar.gz .
	@echo "wrote arxiv-upload.tar.gz:" && tar tzf arxiv-upload.tar.gz

clean:
	latexmk -C && rm -rf $(DIST) arxiv-upload.tar.gz
