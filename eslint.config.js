import js from '@eslint/js';
import tseslint from 'typescript-eslint';
export default tseslint.config(
  { ignores: ['node_modules/**', 'dist/**', 'dist-demo/**'] },
  {
    files: [
      'src/**/*.ts',
      'demo/**/*.ts',
      'examples/**/*.ts',
      'scripts/**/*.ts',
      'tests/**/*.ts',
      '*.config.ts',
    ],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
        },
      ],
    },
  },
);
