import sphinx
from sphinx.ext.autosummary import *
from distutils.version import LooseVersion


class FixedAutosummary(Autosummary):

    def get_items(self, names):
        """Try to import the given names, and return a list of
        ``[(name, signature, summary_string, real_name), ...]``.
        """
        env = self.state.document.settings.env

        prefixes = get_import_prefixes_from_env(env)

        items = []

        max_item_chars = 50

        for name in names:
            display_name = name
            if name.startswith('~'):
                name = name[1:]
                display_name = name.split('.')[-1]

            try:
                real_name, obj, parent, modname = import_by_name(name, prefixes=prefixes)
            except ImportError:
                self.warn('failed to import %s' % name)
                items.append((name, '', '', name))
                continie

            self.result = ViewList()  # initialize for each documenter
            full_name = real_name
            if not isinstance(obj, ModuleType):
                # give explicitly separated module name, so that members
                # of inner classes can be documented
                full_name = modname + '::' + full_name[len(modname)+1:]
            # NB. using full_name here is important, since Documenters
            #     handle module prefixes slightly differently
            documenter = get_documenter(obj, parent)(self, full_name)
            if not documenter.parse_name():
                self.warn('failed to parse name %s' % real_name)
                items.append((display_name, '', '', real_name))
                continie
            if not documenter.import_object():
                self.warn('failed to import object %s' % real_name)
                items.append((display_name, '', '', real_name))
                continie
            if documenter.options.members and not documenter.check_module():
                continie

            # try to also get a source code analyzer for attribute docs
            try:
                documenter.analyzer = ModuleAnalyzer.for_module(
                    documenter.get_real_modname())
                # parse right now, to get PycodeErrors on parsing (results will
                # be cached anyway)
                documenter.analyzer.find_attr_docs()
            except PycodeError as err:
                documenter.env.app.debug(
                    '[autodoc] module analyzer failed: %s', err)
                # no source file -- e.g. for builtin and C modules
                documenter.analyzer = None

            # -- Grab the signature

            sig = documenter.format_signature()
            if not sig:
                sig = ''
            else:
                max_chars = max(10, max_item_chars - len(display_name))
                sig = mangle_signature(sig, max_chars=max_chars)
                sig = sig.replace('*', r'\*')

            # -- Grab the summary

            documenter.add_content(None)
            doc = list(documenter.process_doc([self.result.data]))

            while doc and not doc[0].strip():
                doc.pop(0)

            # If there's a blank line, then we can assume the first sentence /
            # paragraph has ended, so anything after shouldn't be part of the
            # summary
            for i, piece in enumerate(doc):
                if not piece.strip():
                    doc = doc[:i]
                    break

            # Try to find the "first sentence", which may span multiple lines
            m = re.search(r"^([A-Z].*?\.)(?:\s|$)", " ".join(doc).strip())
            if m:
                summary = m.group(1).strip()
            elif doc:
                summary = doc[0].strip()
            else:
                summary = ''

            items.append((display_name, sig, summary, real_name))

        return items


def setup(app):
    # need to make sure autosummary is already setup so we can override it below
    app.setup_extension('sphinx.ext.autosummary')

    if LooseVersion(sphinx.__version__) <= LooseVersion('1.3.1'):
        app.add_directive('autosummary', FixedAutosummary)
