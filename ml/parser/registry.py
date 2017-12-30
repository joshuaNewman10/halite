from ml.parser.starter import StarterParser
from ml.parser.qlearn import QlearnParser

PARSER_REGISTRY = {
    QlearnParser.name: QlearnParser,
    StarterParser.name: StarterParser
}
