from itertools import product

from app.config import cycle_seq, power, symbols


class CodeGen:
    def __init__(self, codelen: int = 8):
        self.codelen = codelen
        self.power = power
        self.alphabet = symbols
        self.cycle_seq = cycle_seq
        if self.cycle_seq[self.codelen] < len(self.alphabet):
            self.logger.error(
                "Слишком малое значение размера двоичного кода, возьмите побольше!"
            )
            raise ValueError(
                "Слишком малое значение размера двоичного кода, возьмите побольше!"
            )
        cyclic_classes = self.generate_cyclic_classes(self.codelen, self.power)
        unique_sequences = sorted(cyclic_classes.keys())
        unique_sequences = sorted(
            unique_sequences, key=lambda x: x.count("1"), reverse=True
        )[: len(self.alphabet)]
        self.patterns = [list(map(int, seq)) for seq in unique_sequences]

    def all_cyclic_shifts(self, sequence):
        """Возвращает все циклические сдвиги последовательности."""
        n = len(sequence)
        return [sequence[i:] + sequence[:i] for i in range(n)]

    def canonical_form(self, sequence):
        """Возвращает каноническую форму последовательности (минимальный циклический сдвиг)."""
        shifts = self.all_cyclic_shifts(sequence)
        return min(shifts)

    def generate_cyclic_classes(self, n=8, alphabet_size=2):
        """Генерирует все уникальные циклические классы для заданной длины и размера алфавита."""
        all_sequences = list(product(range(alphabet_size), repeat=n))
        classes = {}

        for seq in all_sequences:
            seq_str = "".join(map(str, seq))
            canon = self.canonical_form(seq_str)

            if canon not in classes:
                classes[canon] = []
            classes[canon].append(seq_str)
        return classes
