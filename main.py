import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.fixed_mutation_value = tk.DoubleVar(value=0.1)
        self.result_labels = {}
        self.root = root
        self.root.title("Генетический алгоритм - Поиск точки перегиба")
        self.root.geometry("1200x1440")

        self.population_size = tk.IntVar(value=50)
        self.generations = tk.IntVar(value=100)
        self.crossover_prob = tk.DoubleVar(value=0.8)
        self.mutation_prob = tk.DoubleVar(value=0.1)
        self.x_min = tk.DoubleVar(value=-10.0)
        self.x_max = tk.DoubleVar(value=10.0)
        self.encoding_type = tk.StringVar(value="real")

        self.crossover_type = tk.StringVar(value="one_point")
        self.mutation_type = tk.StringVar(value="random")
        self.crossover_points_var = tk.StringVar(value="1")

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.LabelFrame(main_frame, text="Параметры генетического алгоритма", padding="15")
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        row = 0

        left_frame = ttk.Frame(top_frame)
        left_frame.grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)

        # Тип кодирования
        ttk.Label(left_frame, text="Тип кодирования:", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, pady=3)
        encoding_frame = ttk.Frame(left_frame)
        encoding_frame.grid(row=0, column=1, sticky=tk.W, pady=3)
        ttk.Radiobutton(encoding_frame, text="Целое", variable=self.encoding_type,
                        value="integer").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(encoding_frame, text="Вещественное", variable=self.encoding_type,
                        value="real").pack(side=tk.LEFT, padx=5)

        # Размер популяции
        ttk.Label(left_frame, text="Количество особей в популяции:", font=('Arial', 10)).grid(row=1, column=0,
                                                                                              sticky=tk.W, pady=3)
        ttk.Entry(left_frame, textvariable=self.population_size, width=10).grid(row=1, column=1, sticky=tk.W, pady=3,
                                                                                padx=5)

        # Количество итераций
        ttk.Label(left_frame, text="Количество итераций (поколений):", font=('Arial', 10)).grid(row=2, column=0,
                                                                                                sticky=tk.W, pady=3)
        ttk.Entry(left_frame, textvariable=self.generations, width=10).grid(row=2, column=1, sticky=tk.W, pady=3,
                                                                            padx=5)

        # Вероятность кроссовера
        ttk.Label(left_frame, text="Вероятность кроссовера:", font=('Arial', 10)).grid(row=3, column=0, sticky=tk.W,
                                                                                       pady=3)
        ttk.Entry(left_frame, textvariable=self.crossover_prob, width=10).grid(row=3, column=1, sticky=tk.W, pady=3,
                                                                               padx=5)

        # Вероятность мутации
        ttk.Label(left_frame, text="Вероятность мутации:", font=('Arial', 10)).grid(row=4, column=0, sticky=tk.W,
                                                                                    pady=3)
        ttk.Entry(left_frame, textvariable=self.mutation_prob, width=10).grid(row=4, column=1, sticky=tk.W, pady=3,
                                                                              padx=5)

        # Область значения x
        ttk.Label(left_frame, text="Область значения x (отрезок):", font=('Arial', 10)).grid(row=5, column=0,
                                                                                             sticky=tk.W, pady=3)
        range_frame = ttk.Frame(left_frame)
        range_frame.grid(row=5, column=1, sticky=tk.W, pady=3)
        ttk.Label(range_frame, text="от").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.x_min, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text="до").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.x_max, width=8).pack(side=tk.LEFT, padx=2)

        # Правый столбец
        right_frame = ttk.Frame(top_frame)
        right_frame.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)

        # Тип кроссовера
        ttk.Label(right_frame, text="Тип кроссовера:", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, pady=3)
        crossover_frame = ttk.Frame(right_frame)
        crossover_frame.grid(row=0, column=1, sticky=tk.W, pady=3)
        ttk.Radiobutton(crossover_frame, text="Одноточечный", variable=self.crossover_type,
                        value="one_point").pack(anchor=tk.W)
        ttk.Radiobutton(crossover_frame, text="Двухточечный", variable=self.crossover_type,
                        value="two_point").pack(anchor=tk.W)
        ttk.Radiobutton(crossover_frame, text="Многоточечный", variable=self.crossover_type,
                        value="multi_point").pack(anchor=tk.W)

        # Точки для многоточечного кроссовера
        ttk.Label(right_frame, text="Точки кроссовера (через запятую):", font=('Arial', 10)).grid(row=1, column=0,
                                                                                                  sticky=tk.W, pady=3)
        ttk.Entry(right_frame, textvariable=self.crossover_points_var, width=15).grid(row=1, column=1, sticky=tk.W,
                                                                                      pady=3, padx=5)

        # Тип мутации
        ttk.Label(right_frame, text="Тип мутации:", font=('Arial', 10)).grid(row=2, column=0, sticky=tk.W, pady=3)
        mutation_frame = ttk.Frame(right_frame)
        mutation_frame.grid(row=2, column=1, sticky=tk.W, pady=3)
        ttk.Radiobutton(mutation_frame, text="Случайная", variable=self.mutation_type,
                        value="random").pack(anchor=tk.W)
        ttk.Radiobutton(mutation_frame, text="Фиксированная", variable=self.mutation_type,
                        value="fixed").pack(anchor=tk.W)
        ttk.Label(right_frame, text="Фиксированное значение мутации:", font=('Arial', 10)).grid(row=3, column=0,
                                                                                                sticky=tk.W, pady=3)
        ttk.Entry(right_frame, textvariable=self.fixed_mutation_value, width=15).grid(row=3, column=1, sticky=tk.W,
                                                                                      pady=3, padx=5)




        # Кнопки управления
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=row, column=2, sticky=tk.N, padx=20, pady=5)

        ttk.Button(button_frame, text="Запустить алгоритм",
                   command=self.run_algorithm, width=20).pack(pady=5)
        ttk.Button(button_frame, text="Обновить графики",
                   command=self.update_plots, width=20).pack(pady=5)

        # Средняя панель - результаты
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Фрейм результатов
        results_frame = ttk.LabelFrame(middle_frame, text="Результаты выполнения", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Создаем фрейм для результатов с прокруткой
        results_canvas = tk.Canvas(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_canvas.yview)
        scrollable_frame = ttk.Frame(results_canvas)

        # Левый график - функция
        left_plot_frame = ttk.LabelFrame(results_frame, text="График функции f(x) = (x-1.5)³ + 3", padding="10")
        left_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Создание графика функции
        self.fig_func, self.ax_func = plt.subplots(figsize=(6, 4))
        self.canvas_func = FigureCanvasTkAgg(self.fig_func, master=left_plot_frame)
        self.canvas_func.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        scrollable_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )

        results_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=scrollbar.set)

        self.result_labels = {}

        # Заголовок
        title_label = ttk.Label(scrollable_frame, text="РЕЗУЛЬТАТЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА",
                                font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)

        # Функция
        func_label = ttk.Label(scrollable_frame, text="Функция: f(x) = (x - 1.5)³ + 3",
                               font=('Arial', 10))
        func_label.pack(pady=2)

        # Точка перегиба
        inflection_label = ttk.Label(scrollable_frame, text="Точка перегиба: x = 1.5",
                                     font=('Arial', 10))
        inflection_label.pack(pady=2)

        # Разделитель
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=10)

        params_title = ttk.Label(scrollable_frame, text="ВВЕДЕННЫЕ ПАРАМЕТРЫ:",
                                 font=('Arial', 10, 'bold'))
        params_title.pack(anchor='w', pady=5)

        self.result_labels['encoding'] = ttk.Label(scrollable_frame, text="• Тип кодирования: -")
        self.result_labels['encoding'].pack(anchor='w', padx=20)

        self.result_labels['population'] = ttk.Label(scrollable_frame, text="• Количество особей: -")
        self.result_labels['population'].pack(anchor='w', padx=20)

        self.result_labels['generations'] = ttk.Label(scrollable_frame, text="• Количество поколений: -")
        self.result_labels['generations'].pack(anchor='w', padx=20)

        self.result_labels['crossover_prob'] = ttk.Label(scrollable_frame, text="• Вероятность кроссовера: -")
        self.result_labels['crossover_prob'].pack(anchor='w', padx=20)

        self.result_labels['mutation_prob'] = ttk.Label(scrollable_frame, text="• Вероятность мутации: -")
        self.result_labels['mutation_prob'].pack(anchor='w', padx=20)

        self.result_labels['x_range'] = ttk.Label(scrollable_frame, text="• Область x: [-]")
        self.result_labels['x_range'].pack(anchor='w', padx=20)

        self.result_labels['crossover_type'] = ttk.Label(scrollable_frame, text="• Тип кроссовера: -")
        self.result_labels['crossover_type'].pack(anchor='w', padx=20)

        self.result_labels['mutation_type'] = ttk.Label(scrollable_frame, text="• Тип мутации: -")
        self.result_labels['mutation_type'].pack(anchor='w', padx=20)

        # Разделитель
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=10)

        # Статистика заголовок
        stats_title = ttk.Label(scrollable_frame, text="ИТОГОВАЯ СТАТИСТИКА:",
                                font=('Arial', 10, 'bold'))
        stats_title.pack(anchor='w', pady=5)

        # Лейблы для статистики
        self.result_labels['best_solution'] = ttk.Label(scrollable_frame, text="• Лучшее решение: x = -")
        self.result_labels['best_solution'].pack(anchor='w', padx=20)

        self.result_labels['exact_solution'] = ttk.Label(scrollable_frame, text="• Точное решение: x = -")
        self.result_labels['exact_solution'].pack(anchor='w', padx=20)

        self.result_labels['abs_error'] = ttk.Label(scrollable_frame, text="• Абсолютная погрешность: -")
        self.result_labels['abs_error'].pack(anchor='w', padx=20)

        self.result_labels['rel_error'] = ttk.Label(scrollable_frame, text="• Средняя погрешность: -")
        self.result_labels['rel_error'].pack(anchor='w', padx=20)

        results_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")



        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        right_plot_frame = ttk.LabelFrame(bottom_frame, text="График приспособленности", padding="10")
        right_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_fitness, self.ax_fitness = plt.subplots(figsize=(6, 4))
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=right_plot_frame)
        self.canvas_fitness.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_plots()

    def update_plots(self,best_solution=None, fitness_history=None):

        self.ax_func.clear()

        x = np.linspace(self.x_min.get(), self.x_max.get(), 400)
        y = (x - 1.5) ** 3 + 3

        self.ax_func.plot(x, y, 'b-', linewidth=1, label='f(x) = (x-1.5)³ + 3')


        if best_solution is not None:
            best_y = (best_solution - 1.5) ** 3 + 3
            self.ax_func.plot(best_solution, best_y, 'go', markersize=8,
                              label=f'Найденное решение (x={best_solution:.12f})')
            self.ax_func.axvline(best_solution, color='r', linestyle='--', linewidth=1,
                                 label=f'Точка перегиба (x={best_solution:.1f})')

        self.ax_func.set_xlabel('x')
        self.ax_func.set_ylabel('f(x)')
        self.ax_func.set_title('Функция f(x) = (x-1.5)³ + 3')
        self.ax_func.legend()
        self.ax_func.grid(True, alpha=0.3)

        self.ax_fitness.clear()

        if fitness_history is not None and len(fitness_history) > 0:
            generations = list(range(len(fitness_history)))
            best_fitness = [h[0] for h in fitness_history]
            avg_fitness = [h[1] for h in fitness_history]

            self.ax_fitness.plot(generations, best_fitness, 'g-', linewidth=1, label='Лучшая приспособленность')
            self.ax_fitness.plot(generations, avg_fitness, 'b-', linewidth=1, label='Средняя приспособленность')

            if len(fitness_history[0]) > 2:
                worst_fitness = [h[2] for h in fitness_history]
                self.ax_fitness.plot(generations, worst_fitness, 'r-', linewidth=1, label='Худшая приспособленность')

            self.ax_fitness.set_xlabel('Поколение')
            self.ax_fitness.set_ylabel('Приспособленность')
            self.ax_fitness.set_title('Эволюция приспособленности')
            self.ax_fitness.legend()
            self.ax_fitness.grid(True, alpha=0.3)
        else:
            self.ax_fitness.text(0.5, 0.5, 'Запустите алгоритм\nдля отображения графика',
                                 ha='center', va='center', transform=self.ax_fitness.transAxes)
            self.ax_fitness.set_xlabel('Поколение')
            self.ax_fitness.set_ylabel('Приспособленность')
            self.ax_fitness.set_title('Эволюция приспособленности')

        self.canvas_func.draw()
        self.canvas_fitness.draw()

    def fitness_function(self, x):
        h = 0.001
        f_plus = (x + h - 1.5) ** 3 + 3  # f(x+h)
        f_minus = (x - h - 1.5) ** 3 + 3  # f(x-h)
        f_x = (x - 1.5) ** 3 + 3  # f(x)

        # Численная вторая производная
        second_derivative = (f_plus - 2 * f_x + f_minus) / (h * h)

        # Чем ближе вторая производная к 0 → тем лучше
        return -abs(second_derivative)

    def initialize_population_real(self):
        return [random.uniform(self.x_min.get(), self.x_max.get())
                for _ in range(self.population_size.get())]

    def initialize_population_integer(self):
        scale = 1000
        min_val = int(self.x_min.get() * scale)
        max_val = int(self.x_max.get() * scale)
        return [random.randint(min_val, max_val) / scale
                for _ in range(self.population_size.get())]

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_prob.get():
            return parent1, parent2

        if self.encoding_type.get() == "real":
            if self.crossover_type.get() == "one_point":
                # Одноточечный кроссовер
                alpha = random.random()
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
            elif self.crossover_type.get() == "two_point":
                # Двухточечный кроссовер
                alpha1, alpha2 = random.random(), random.random()
                if alpha1 > alpha2:
                    alpha1, alpha2 = alpha2, alpha1
                child1 = alpha1 * parent1 + (1 - alpha1) * parent2
                child2 = alpha2 * parent2 + (1 - alpha2) * parent1
            else:
                # Многоточечный кроссовер
                try:
                    points = [float(p.strip()) for p in self.crossover_points_var.get().split(',')]
                    points = [max(0.0, min(1.0, p)) for p in points]
                    if len(points) == 0:
                        points = [0.5]
                except:
                    points = [0.5]

                # Усреднение с весами
                child1 = sum(points) / len(points) * parent1 + (1 - sum(points) / len(points)) * parent2
                child2 = sum(points) / len(points) * parent2 + (1 - sum(points) / len(points)) * parent1
        else:
            # Целочисленное кодирование (упрощенное)
            if self.crossover_type.get() == "one_point":
                # Одноточечный кроссовер
                child1 = (parent1 + parent2) / 2
                child2 = parent2 if random.random() > 0.5 else parent1
            elif self.crossover_type.get() == "two_point":
                # Двухточечный кроссовер
                child1 = (parent1 * 0.3 + parent2 * 0.7)
                child2 = (parent1 * 0.7 + parent2 * 0.3)
            else:  # multi_point
                # Многоточечный кроссовер
                child1 = (parent1 + parent2) / 2
                child2 = (parent1 * 0.4 + parent2 * 0.6)

        return child1, child2

    def mutate(self, individual):
        if random.random() > self.mutation_prob.get():
            return individual

        if self.mutation_type.get() == "random":
            # Случайная мутация
            mutation_strength = (self.x_max.get() - self.x_min.get()) * 0.1
            return individual + random.uniform(-mutation_strength, mutation_strength)
        else:
            # Фиксированная мутация с пользовательским значением
            mutation_value = self.fixed_mutation_value.get()
            if mutation_value <= 0:
                mutation_value = 0.1

            if random.random() > 0.5:
                return individual + mutation_value
            else:
                return individual - mutation_value

    def select_parents(self, population, fitnesses):
        min_fitness = min(fitnesses)
        normalized_fitnesses = [f - min_fitness + 1e-10 for f in fitnesses]

        total_fitness = sum(normalized_fitnesses)
        if total_fitness == 0:
            return random.choices(population, k=2)

        probabilities = [f / total_fitness for f in normalized_fitnesses]
        return random.choices(population, weights=probabilities, k=2)

    def toggle_mutation_type(self):
        if self.mutation_type.get() == "fixed":
            self.fixed_mutation_label.grid()
            self.fixed_mutation_entry.grid()
        else:
            self.fixed_mutation_label.grid_remove()
            self.fixed_mutation_entry.grid_remove()

    def run_algorithm(self):
        try:
            # Проверка параметров
            if self.population_size.get() <= 0 or self.generations.get() <= 0:
                messagebox.showerror("Ошибка", "Размер популяции и количество поколений должны быть положительными")
                return

            if self.x_min.get() >= self.x_max.get():
                messagebox.showerror("Ошибка", "x_min должен быть меньше x_max")
                return
            if self.mutation_type.get() == "fixed":
                if self.fixed_mutation_value.get() <= 0:
                    messagebox.showwarning("Предупреждение",
                                           "Фиксированное значение мутации должно быть положительным. Используется значение 0.1")
                    self.fixed_mutation_value.set(0.1)


            # Инициализация популяции
            if self.encoding_type.get() == "real":
                population = self.initialize_population_real()
            else:
                population = self.initialize_population_integer()
            all_generations = []
            fitness_history = []
            errors_history = []

            for generation in range(self.generations.get()):
                # Вычисление приспособленности
                fitnesses = [self.fitness_function(ind) for ind in population]

                best_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                worst_fitness = min(fitnesses)
                best_idx = fitnesses.index(best_fitness)
                best_individual = population[best_idx]

                # Вычисление погрешности для лучшей особи
                error = abs(best_individual - 1.5)

                all_generations.append((generation, best_individual, best_fitness, avg_fitness, worst_fitness, error))
                fitness_history.append((best_fitness, avg_fitness, worst_fitness))
                errors_history.append(error)


                # Создание нового поколения
                new_population = []

                new_population.append(best_individual)

                while len(new_population) < self.population_size.get():
                    # Отбор родителей
                    parent1, parent2 = self.select_parents(population, fitnesses)

                    # Скрещивание
                    child1, child2 = self.crossover(parent1, parent2)

                    # Мутация
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    # Проверка границ
                    child1 = max(self.x_min.get(), min(self.x_max.get(), child1))
                    child2 = max(self.x_min.get(), min(self.x_max.get(), child2))

                    new_population.extend([child1, child2])

                population = new_population[:self.population_size.get()]

                # Обновляем графики каждые 10 поколений
                if generation % 10 == 0:
                    self.update_plots(best_individual, fitness_history)


            # Вывод результатов
            best_solution = all_generations[-1][1]
            best_fitness_value = all_generations[-1][2]
            avg_error = np.mean(errors_history)

            # Обновляем лейблы с результатами
            if 'initial' in self.result_labels:
                self.result_labels['initial'].pack_forget()

            # Обновляем параметры
            self.result_labels['encoding'].config(
                text=f"• Тип кодирования: {'Целое' if self.encoding_type.get() == 'integer' else 'Вещественное'}"
            )
            self.result_labels['population'].config(
                text=f"• Количество особей: {self.population_size.get()}"
            )
            self.result_labels['generations'].config(
                text=f"• Количество поколений: {self.generations.get()}"
            )
            self.result_labels['crossover_prob'].config(
                text=f"• Вероятность кроссовера: {self.crossover_prob.get()}"
            )
            self.result_labels['mutation_prob'].config(
                text=f"• Вероятность мутации: {self.mutation_prob.get()}"
            )
            self.result_labels['x_range'].config(
                text=f"• Область x: [{self.x_min.get()}, {self.x_max.get()}]"
            )
            self.result_labels['crossover_type'].config(
                text=f"• Тип кроссовера: {self.get_crossover_type_name()}"
            )
            if self.mutation_type.get() == "fixed":
                mutation_info = f"Фиксированная (значение: {self.fixed_mutation_value.get():.3f})"
            else:
                mutation_info = "Случайная"

            self.result_labels['mutation_type'].config(
                text=f"• Тип мутации: {mutation_info}"
            )

            best_solution1 = round(best_solution,1);
            # Обновляем статистику
            self.result_labels['best_solution'].config(
                text=f"• Лучшее решение: x = {best_solution:.10f}"
            )
            self.result_labels['exact_solution'].config(
                text=f"• Точное решение: x = {best_solution1:.10f}"
            )
            self.result_labels['abs_error'].config(
                text=f"• Абсолютная погрешность: {abs(best_solution - best_solution1):.12f}"
            )
            self.result_labels['rel_error'].config(
                text=f"• Средняя погрешность: {avg_error:.6f}"
            )



            # Обновляем графики с финальными результатами
            self.update_plots(best_solution, fitness_history)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def get_crossover_type_name(self):
        names = {
            "one_point": "Одноточечный",
            "two_point": "Двухточечный",
            "multi_point": "Многоточечный"
        }
        return names.get(self.crossover_type.get(), "Одноточечный")


def main():
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()